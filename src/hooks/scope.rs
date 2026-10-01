//! Scoped authorizers: temporarily tighten a `Connection`'s authorizer and
//! automatically restore the previous one when the scope ends.

use std::collections::BTreeSet;
use std::fmt;

use crate::hooks::{AuthAction, AuthContext, Authorization};
use crate::{Connection, Result};

/// A policy deciding whether a statement action is authorized inside an
/// `AuthorizerScope`.
///
/// Implemented for all closures `Fn(AuthContext<'_>) -> Authorization`, and
/// for the ready-made `ReadOnlyTables` policy.
pub trait ScopePolicy: Send + 'static {
    /// Return the `Authorization` verdict for the action described by
    /// `ctx`. Returning `Authorization::Deny` or
    /// `Authorization::Ignore` tightens the scope; the effective verdict
    /// is the most restrictive one of all active hooks.
    fn authorize(&self, ctx: AuthContext<'_>) -> Authorization;
}

impl<F> ScopePolicy for F
where
    F: for<'r> Fn(AuthContext<'r>) -> Authorization + Send + 'static,
{
    #[inline]
    fn authorize(&self, ctx: AuthContext<'_>) -> Authorization {
        self(ctx)
    }
}

/// RAII guard returned by `Connection::authorizer_scope`.
///
/// While the guard is alive, the connection's authorizer is tightened with
/// the scope's policy. When the guard is dropped -- normally at the end of
/// the enclosing block, or while unwinding from a panic -- the previous
/// authorizer (a base hook registered with `Connection::authorizer`, an
/// outer scope, or none at all) is restored.
///
/// The guard borrows the connection immutably, so statements (even cached
/// ones) may still be prepared and executed while the scope is active, and
/// statements that are still alive when the scope is entered or left cannot
/// be used to bypass it: the prepared-statement cache is flushed on every
/// transition and statements prepared under a different authorizer
/// generation are never reused.
///
/// Scopes may be nested; nested scopes compute the intersection of their
/// restrictions (an action is allowed only if every active scope allows it).
#[must_use = "the scope ends when the guard is dropped"]
pub struct AuthorizerScope<'conn> {
    conn: &'conn Connection,
}

impl fmt::Debug for AuthorizerScope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthorizerScope").finish_non_exhaustive()
    }
}

impl Drop for AuthorizerScope<'_> {
    fn drop(&mut self) {
        // Restore the previous authorizer, even while unwinding from a
        // panic. No `RefCell` borrow of the inner connection is held
        // across statement execution, so this cannot conflict with an
        // active borrow.
        let mut inner = self.conn.db.borrow_mut();
        if let Err(e) = inner.pop_scoped_authorizer() {
            debug_assert!(false, "failed to restore authorizer: {e}");
            drop(e);
        }
        drop(inner);
        self.conn.bump_authorizer_generation();
    }
}

impl Connection {
    /// Enter a scope in which the connection's authorizer is tightened with
    /// `policy`.
    ///
    /// The policy is combined with the currently active authorizer (if any):
    /// an action is only performed if both allow it. Dropping the returned
    /// `AuthorizerScope` guard -- including during panic unwinding --
    /// restores the previous authorizer. Entering and leaving the scope
    /// invalidates the prepared-statement cache, so statements prepared
    /// outside the scope cannot bypass the policy.
    ///
    /// # Example
    ///
    /// ```
    /// use rusqlite::hooks::ReadOnlyTables;
    /// use rusqlite::{Connection, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let db = Connection::open_in_memory()?;
    ///     db.execute_batch("CREATE TABLE public(id INTEGER); CREATE TABLE private(id INTEGER);")?;
    ///     {
    ///         let _scope = db.authorizer_scope(ReadOnlyTables::new().allow_table("public"))?;
    ///         db.execute_batch("SELECT id FROM public")?; // allowed
    ///         assert!(db.execute_batch("SELECT id FROM private").is_err()); // denied
    ///         assert!(db.execute_batch("INSERT INTO public VALUES (1)").is_err()); // denied
    ///     }
    ///     db.execute_batch("INSERT INTO public VALUES (1)")?; // restored: writable again
    ///     Ok(())
    /// }
    /// ```
    pub fn authorizer_scope<P>(&self, policy: P) -> Result<AuthorizerScope<'_>>
    where
        P: ScopePolicy,
    {
        self.db
            .borrow_mut()
            .push_scoped_authorizer(Box::new(move |ctx| policy.authorize(ctx)))?;
        self.bump_authorizer_generation();
        Ok(AuthorizerScope { conn: self })
    }
}

/// A read-only table whitelist policy for `Connection::authorizer_scope`.
///
/// The policy inspects the actions SQLite reports to the authorizer while a
/// statement is being prepared -- it never guesses from the SQL text (so
/// e.g. writes hidden behind CTEs cannot slip through, and reads are
/// checked against the *actual* tables being read). Because SQLite reports
/// the underlying table together with the inner-most view or trigger
/// responsible for an access, accesses performed through views and triggers
/// are checked exactly like direct ones.
///
/// By default the policy:
///
/// * allows `SELECT` reads of whitelisted tables only (reading through a
///   view requires whitelisting the view *and* its underlying tables:
///   SQLite reports reads of the view's output columns under the view's
///   name, and reads of the real columns under the underlying table's
///   name);
/// * denies every write (`INSERT`/`UPDATE`/`DELETE`), schema change
///   (`CREATE`/`DROP`/`ALTER`/`REINDEX`/`ANALYZE`/virtual tables)
///   and `ATTACH`/`DETACH`;
/// * denies `PRAGMA`s except a small set of read-only ones (see
///   `allow_pragma`); state-changing pragmas such as
///   `writable_schema`, `journal_mode` or `foreign_keys` are always
///   denied unless explicitly whitelisted;
/// * denies SQL function calls except a set of pure, read-only built-ins
///   (see `allow_function`); dangerous functions such as
///   `load_extension`, `readfile` or `writefile` are never allowed by
///   default;
/// * allows transaction control
///   (`BEGIN`/`COMMIT`/`ROLLBACK`/`SAVEPOINT`), which by itself
///   cannot modify data.
///
/// # Example
///
/// ```
/// use rusqlite::hooks::ReadOnlyTables;
/// use rusqlite::{Connection, Result};
///
/// fn main() -> Result<()> {
///     let db = Connection::open_in_memory()?;
///     db.execute_batch("CREATE TABLE t(x); CREATE VIEW v AS SELECT x FROM t;")?;
///     let scope = db.authorizer_scope(
///         ReadOnlyTables::new()
///             .allow_table("t")
///             .allow_table("v") // views are checked by name as well
///             .allow_function("length"),
///     )?;
///     db.execute_batch("SELECT length(x) FROM v")?; // view reads resolve to `t`: allowed
///     drop(scope);
///     Ok(())
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct ReadOnlyTables {
    tables: BTreeSet<String>,
    functions: BTreeSet<String>,
    pragmas: BTreeSet<String>,
}

impl ReadOnlyTables {
    /// A policy allowing reads of whitelisted tables, a default set of pure
    /// read-only built-in functions and a default set of read-only pragmas.
    /// Everything else is denied.
    pub fn new() -> Self {
        let mut policy = Self::empty();
        // Pure, read-only built-in functions. Notably absent:
        // `load_extension`, `readfile`, `writefile` and anything else
        // with side effects.
        policy.functions.extend(
            [
                "abs",
                "acos",
                "acosh",
                "asin",
                "asinh",
                "atan",
                "atan2",
                "atanh",
                "avg",
                "ceil",
                "ceiling",
                "char",
                "coalesce",
                "concat",
                "concat_ws",
                "cos",
                "cosh",
                "count",
                "date",
                "datetime",
                "degrees",
                "exp",
                "floor",
                "format",
                "glob",
                "group_concat",
                "hex",
                "ifnull",
                "iif",
                "instr",
                "julianday",
                "last_insert_rowid",
                "length",
                "like",
                "likelihood",
                "likely",
                "ln",
                "log",
                "log10",
                "log2",
                "lower",
                "ltrim",
                "max",
                "min",
                "mod",
                "nullif",
                "octet_length",
                "pi",
                "pow",
                "power",
                "printf",
                "quote",
                "radians",
                "random",
                "replace",
                "round",
                "rtrim",
                "sign",
                "sin",
                "sinh",
                "soundex",
                "sqlite_source_id",
                "sqlite_version",
                "sqrt",
                "strftime",
                "string_agg",
                "substr",
                "substring",
                "sum",
                "tan",
                "tanh",
                "time",
                "timediff",
                "total",
                "trim",
                "trunc",
                "typeof",
                "unicode",
                "unhex",
                "unixepoch",
                "unlikely",
                "upper",
                "zeroblob",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        // Read-only pragmas.
        policy.pragmas.extend(
            [
                "compile_options",
                "database_list",
                "foreign_key_list",
                "index_info",
                "index_list",
                "integrity_check",
                "quick_check",
                "table_info",
                "table_list",
                "table_xinfo",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        policy
    }

    /// A policy that denies everything not explicitly allowed: no default
    /// functions or pragmas are permitted.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Allow `SELECT` reads of `table` (case-insensitive). Reads
    /// performed through views or triggers are also checked against the
    /// underlying table, so whitelisting a view name alone does not grant
    /// access to the tables it reads.
    pub fn allow_table(mut self, table: &str) -> Self {
        self.tables.insert(table.to_lowercase());
        self
    }

    /// Allow `SELECT` reads of every table in `tables`.
    pub fn allow_tables<'a>(mut self, tables: impl IntoIterator<Item = &'a str>) -> Self {
        self.tables
            .extend(tables.into_iter().map(str::to_lowercase));
        self
    }

    /// Additionally allow calls to the SQL function `function`
    /// (case-insensitive). Only add functions known to be free of side
    /// effects.
    pub fn allow_function(mut self, function: &str) -> Self {
        self.functions.insert(function.to_lowercase());
        self
    }

    /// Additionally allow the `PRAGMA` named `pragma`
    /// (case-insensitive). Only add pragmas known to be read-only.
    pub fn allow_pragma(mut self, pragma: &str) -> Self {
        self.pragmas.insert(pragma.to_lowercase());
        self
    }
}

impl ScopePolicy for ReadOnlyTables {
    fn authorize(&self, ctx: AuthContext<'_>) -> Authorization {
        match ctx.action {
            // Transaction control cannot modify data by itself; every actual
            // access is checked separately.
            AuthAction::Select
            | AuthAction::Transaction { .. }
            | AuthAction::Savepoint { .. }
            | AuthAction::Recursive => Authorization::Allow,
            // SQLite reports the underlying table even when the access is
            // performed through a view or a trigger (`ctx.accessor` then
            // names that view or trigger), so checking the table name covers
            // indirect accesses too.
            AuthAction::Read { table_name, .. } => {
                if self.tables.contains(&table_name.to_lowercase()) {
                    Authorization::Allow
                } else {
                    Authorization::Deny
                }
            }
            AuthAction::Function { function_name } => {
                if self.functions.contains(&function_name.to_lowercase()) {
                    Authorization::Allow
                } else {
                    Authorization::Deny
                }
            }
            AuthAction::Pragma { pragma_name, .. } => {
                if self.pragmas.contains(&pragma_name.to_lowercase()) {
                    Authorization::Allow
                } else {
                    Authorization::Deny
                }
            }
            // Writes, schema changes, ATTACH/DETACH and anything unknown
            // are denied.
            _ => Authorization::Deny,
        }
    }
}
