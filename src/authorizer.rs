//! Scoped, composable authorizers for `Connection`.
//!
//! This module builds on top of the raw authorizer hook
//! (`Connection::authorizer`) and provides:
//!
//! - `Connection::authorizer_scope`: temporarily tightens the connection
//!   authorizer for the duration of a lexical scope. The previous
//!   authorization state is restored when the returned `AuthorizerScope`
//!   guard is dropped, including during panic unwinding.
//! - `ReadOnlyWhitelist`: a ready-made `AuthorizerPolicy` that only
//!   allows reads on an explicit whitelist of tables, rejects writes, DDL,
//!   `ATTACH`, dangerous `PRAGMA`s and unauthorized functions.
//!
//! Authorization decisions are made from the SQLite authorizer action codes
//! (see `AuthAction`), never by inspecting the SQL text, so views and
//! triggers cannot smuggle access to underlying tables: SQLite reports the
//! actual table being read or written (with `AuthContext::accessor` naming
//! the responsible view or trigger), and the policy decides based on that.
//!
//! Nested scopes are combined as an intersection of their restrictions: a
//! statement is only authorized if *every* active layer allows it.
//!
//! The prepared-statement cache is flushed whenever a scope is entered or
//! exited, and statements checked out before the transition are discarded
//! rather than returned to the cache, so a statement prepared under a looser
//! policy cannot be replayed under a stricter one (and vice versa).

use std::collections::BTreeSet;

use crate::hooks::{AuthAction, AuthContext, Authorization};
use crate::{Connection, Result};

/// A policy that decides whether a single SQLite action is authorized.
///
/// Any closure `FnMut(AuthContext<'_>) -> Authorization` automatically
/// implements this trait; `ReadOnlyWhitelist` is a ready-made
/// implementation.
pub trait AuthorizerPolicy: Send + 'static {
    /// Decide whether the action described by `ctx` is authorized.
    fn authorize(&mut self, ctx: AuthContext<'_>) -> Authorization;
}

impl<F> AuthorizerPolicy for F
where
    F: for<'r> FnMut(AuthContext<'r>) -> Authorization + Send + 'static,
{
    #[inline]
    fn authorize(&mut self, ctx: AuthContext<'_>) -> Authorization {
        self(ctx)
    }
}

/// Well-known read-only SQLite built-in functions allowed by default.
const DEFAULT_ALLOWED_FUNCTIONS: &[&str] = &[
    "abs",
    "avg",
    "char",
    "coalesce",
    "concat",
    "concat_ws",
    "count",
    "date",
    "datetime",
    "format",
    "glob",
    "group_concat",
    "hex",
    "ifnull",
    "iif",
    "instr",
    "julianday",
    "length",
    "like",
    "lower",
    "ltrim",
    "max",
    "min",
    "nullif",
    "octet_length",
    "printf",
    "quote",
    "random",
    "replace",
    "round",
    "rtrim",
    "sign",
    "string_agg",
    "strftime",
    "substr",
    "substring",
    "sum",
    "time",
    "total",
    "trim",
    "typeof",
    "unicode",
    "unixepoch",
    "upper",
    "zeroblob",
];

/// `PRAGMA`s that are denied even when used without a value, because
/// they mutate database or connection state as a side effect.
const DEFAULT_DENIED_PRAGMAS: &[&str] = &[
    "writable_schema",
    "trusted_schema",
    "wal_checkpoint",
    "optimize",
    "shrink_memory",
];

/// A read-only `AuthorizerPolicy` based on a whitelist of table names.
///
/// The policy:
///
/// - allows `SELECT` reads whose *actual* table (as reported by the
///   SQLite authorizer, so including tables behind views and triggers) is in
///   the whitelist; table names are matched case-insensitively;
///   note that SQLite reports reads on a view under the view's own name, so
///   a view you want to expose must be whitelisted like a table — reads on
///   its underlying tables are still checked separately;
/// - denies every write and DDL action (insert, update, delete, create,
///   drop, alter, reindex, analyze, ...);
/// - denies `ATTACH` and `DETACH`;
/// - denies `PRAGMA` setters (any pragma used with a value) and a
///   denylist of dangerous pragmas (see `ReadOnlyWhitelist::deny_pragma`);
/// - denies SQL functions that are not explicitly allowed; a curated set of
///   read-only built-ins is allowed by default (see
///   `ReadOnlyWhitelist::allow_function`).
///
/// Anything not explicitly allowed is denied (fail closed).
#[derive(Clone, Debug)]
pub struct ReadOnlyWhitelist {
    tables: BTreeSet<String>,
    functions: BTreeSet<String>,
    denied_pragmas: BTreeSet<String>,
}

impl Default for ReadOnlyWhitelist {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ReadOnlyWhitelist {
    /// Create a policy that allows no table at all; add tables with
    /// `allow_table`(Self::allow_table).
    pub fn new() -> Self {
        Self {
            tables: BTreeSet::new(),
            functions: DEFAULT_ALLOWED_FUNCTIONS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            denied_pragmas: DEFAULT_DENIED_PRAGMAS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }

    /// Allow reading `table` (case-insensitive).
    pub fn allow_table(mut self, table: &str) -> Self {
        self.tables.insert(table.to_lowercase());
        self
    }

    /// Allow reading each of `tables` (case-insensitive).
    pub fn allow_tables<I, S>(mut self, tables: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.tables
            .extend(tables.into_iter().map(|t| t.as_ref().to_lowercase()));
        self
    }

    /// Allow calling the SQL function `function` (case-insensitive).
    pub fn allow_function(mut self, function: &str) -> Self {
        self.functions.insert(function.to_lowercase());
        self
    }

    /// Additionally deny the `PRAGMA` named `pragma`, even
    /// when used without a value (case-insensitive).
    pub fn deny_pragma(mut self, pragma: &str) -> Self {
        self.denied_pragmas.insert(pragma.to_lowercase());
        self
    }
}

impl AuthorizerPolicy for ReadOnlyWhitelist {
    fn authorize(&mut self, ctx: AuthContext<'_>) -> Authorization {
        match ctx.action {
            // Pure query / transaction-control actions are always fine; the
            // individual table accesses they cause are checked separately.
            AuthAction::Select
            | AuthAction::Transaction { .. }
            | AuthAction::Savepoint { .. }
            | AuthAction::Recursive => Authorization::Allow,
            AuthAction::Read { table_name, .. } => {
                if self.tables.contains(table_name.to_lowercase().as_str()) {
                    Authorization::Allow
                } else {
                    Authorization::Deny
                }
            }
            AuthAction::Function { function_name } => {
                if self
                    .functions
                    .contains(function_name.to_lowercase().as_str())
                {
                    Authorization::Allow
                } else {
                    Authorization::Deny
                }
            }
            AuthAction::Pragma {
                pragma_name,
                pragma_value,
            } => {
                if pragma_value.is_some()
                    || self
                        .denied_pragmas
                        .contains(pragma_name.to_lowercase().as_str())
                {
                    Authorization::Deny
                } else {
                    Authorization::Allow
                }
            }
            // Everything else (writes, DDL, ATTACH/DETACH, unknown action
            // codes) is denied.
            _ => Authorization::Deny,
        }
    }
}

/// RAII guard returned by `Connection::authorizer_scope`.
///
/// While the guard is alive, its policy is layered on top of whatever
/// authorizer was previously installed on the connection (the most
/// restrictive decision wins). Dropping the guard — normally at the end of
/// the enclosing scope, or during panic unwinding — removes the layer and
/// restores the previous authorization state.
///
/// The guard borrows the connection (`&'conn Connection`), so it can
/// never outlive it, and nested guards created with
/// `AuthorizerScope::nested` cannot outlive their parent scope.
#[derive(Debug)]
pub struct AuthorizerScope<'conn> {
    conn: &'conn Connection,
    layer_id: u64,
}

impl<'conn> AuthorizerScope<'conn> {
    /// Enter a nested scope, further restricting authorization until the
    /// returned guard is dropped. The effective policy is the intersection
    /// of all enclosing scopes and the base authorizer.
    pub fn nested<P: AuthorizerPolicy>(&self, policy: P) -> Result<AuthorizerScope<'_>> {
        self.conn.authorizer_scope(policy)
    }

    /// The connection this scope is attached to.
    #[inline]
    pub fn connection(&self) -> &'conn Connection {
        self.conn
    }
}

impl Drop for AuthorizerScope<'_> {
    fn drop(&mut self) {
        // Fail closed: if the connection is unexpectedly borrowed (which
        // cannot happen in normal use), leave the restrictive layer in place
        // rather than panicking while possibly unwinding.
        if let Ok(mut db) = self.conn.db.try_borrow_mut() {
            let _ = db.pop_authorizer_scope(self.layer_id);
        }
        #[cfg(feature = "cache")]
        self.conn.flush_prepared_statement_cache();
    }
}

impl Connection {
    /// Temporarily tighten this connection's authorizer with `policy`
    /// until the returned `AuthorizerScope` guard is dropped.
    ///
    /// The policy is combined with the authorizer previously installed via
    /// `Connection::authorizer` (if any) and with any enclosing scopes:
    /// every layer is consulted and the most restrictive decision wins, so a
    /// scope can only ever *narrow* what is allowed. Dropping the guard —
    /// including during panic unwinding — restores the previous state.
    ///
    /// The prepared-statement cache is flushed on entry and on exit so that
    /// statements prepared under a different policy cannot be reused to
    /// bypass the current one.
    ///
    /// ```rust
    /// use rusqlite::authorizer::ReadOnlyWhitelist;
    /// use rusqlite::{Connection, Result};
    ///
    /// fn main() -> Result<()> {
    ///     let conn = Connection::open_in_memory()?;
    ///     conn.execute_batch("CREATE TABLE public (v TEXT); INSERT INTO public VALUES ('a');")?;
    ///     {
    ///         let _scope =
    ///             conn.authorizer_scope(ReadOnlyWhitelist::new().allow_table("public"))?;
    ///         let n: i64 = conn.query_row("SELECT count(*) FROM public", [], |r| r.get(0))?;
    ///         assert_eq!(n, 1);
    ///         // Writes are rejected while the scope is active...
    ///         assert!(conn.execute("INSERT INTO public VALUES ('b')", []).is_err());
    ///     }
    ///     // ... and allowed again once it is dropped.
    ///     conn.execute("INSERT INTO public VALUES ('b')", [])?;
    ///     Ok(())
    /// }
    /// ```
    pub fn authorizer_scope<P: AuthorizerPolicy>(
        &self,
        mut policy: P,
    ) -> Result<AuthorizerScope<'_>> {
        let layer_id = self
            .db
            .borrow_mut()
            .push_authorizer_scope(move |ctx: AuthContext<'_>| policy.authorize(ctx))?;
        #[cfg(feature = "cache")]
        self.flush_prepared_statement_cache();
        Ok(AuthorizerScope {
            conn: self,
            layer_id,
        })
    }
}

#[cfg(all(test, not(miri)))]
mod test {
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    use super::ReadOnlyWhitelist;
    use crate::hooks::{AuthAction, AuthContext, Authorization};
    use crate::{Connection, Result};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn setup() -> Result<Connection> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "
            CREATE TABLE public (id INTEGER PRIMARY KEY, v TEXT);
            CREATE TABLE semi (id INTEGER PRIMARY KEY, v TEXT);
            CREATE TABLE private (id INTEGER PRIMARY KEY, secret TEXT);
            INSERT INTO public VALUES (1, 'a');
            INSERT INTO semi VALUES (1, 's');
            INSERT INTO private VALUES (1, 'top-secret');
            CREATE VIEW v_public AS SELECT id, v FROM public;
            CREATE VIEW v_private AS SELECT id, secret FROM private;
            CREATE TRIGGER trg_leak AFTER INSERT ON public BEGIN
                INSERT INTO private (id, secret) VALUES (NEW.id, 'leak');
            END;
            ",
        )?;
        Ok(conn)
    }

    fn counting_authorizer(
        calls: Arc<AtomicUsize>,
    ) -> impl FnMut(AuthContext<'_>) -> Authorization {
        move |_: AuthContext<'_>| {
            calls.fetch_add(1, Ordering::SeqCst);
            Authorization::Allow
        }
    }

    #[test]
    fn readonly_scope_allows_denies_and_restores() -> Result<()> {
        let mut conn = setup()?;
        let calls = Arc::new(AtomicUsize::new(0));

        // Writable outside any scope.
        conn.execute("INSERT INTO public (id, v) VALUES (2, 'b')", [])?;
        println!("允许: 作用域外可写 public");

        // Install a base authorizer that allows everything and counts calls.
        conn.authorizer(Some(counting_authorizer(Arc::clone(&calls))))?;
        let before_scope = calls.load(Ordering::SeqCst);

        {
            let _scope = conn
                .authorizer_scope(ReadOnlyWhitelist::new().allow_tables(["public", "v_public"]))?;

            // Reads on the whitelisted table are allowed, directly and via a view.
            let n: i64 = conn.query_row("SELECT count(*) FROM public", [], |r| r.get(0))?;
            let v: String =
                conn.query_row("SELECT v FROM v_public WHERE id = 1", [], |r| r.get(0))?;
            println!("允许: 作用域内读 public (count={n}, view={v})");
            // The base authorizer is still consulted inside the scope.
            assert!(calls.load(Ordering::SeqCst) > before_scope);

            // Writes are denied.
            assert!(
                conn.execute("INSERT INTO public (id, v) VALUES (3, 'c')", [])
                    .is_err()
            );
            assert!(conn.execute("DELETE FROM public", []).is_err());
            println!("拒绝: 作用域内写 public");

            // Reads on non-whitelisted tables are denied, directly and via a view.
            assert!(
                conn.query_row("SELECT secret FROM private", [], |r| r.get::<_, String>(0))
                    .is_err()
            );
            assert!(
                conn.query_row("SELECT secret FROM v_private", [], |r| r
                    .get::<_, String>(0))
                    .is_err()
            );
            println!("拒绝: 作用域内直接/经 view 读 private");

            // ATTACH and dangerous PRAGMAs are denied.
            assert!(
                conn.execute_batch("ATTACH DATABASE ':memory:' AS evil")
                    .is_err()
            );
            assert!(conn.execute_batch("PRAGMA writable_schema = 1").is_err());
            assert!(
                conn.query_row("PRAGMA writable_schema", [], |r| r.get::<_, i64>(0))
                    .is_err()
            );
            // A harmless read-only pragma still works.
            let _: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
            println!("拒绝: ATTACH 与危险 PRAGMA; 允许: 只读 PRAGMA user_version");
        }

        // The previous authorizer is restored on scope exit.
        conn.execute("INSERT INTO public (id, v) VALUES (3, 'c')", [])?;
        let restored = calls.load(Ordering::SeqCst);
        let s: String = conn.query_row("SELECT secret FROM private", [], |r| r.get(0))?;
        assert_eq!(s, "top-secret");
        assert!(calls.load(Ordering::SeqCst) > restored);
        println!("恢复: 退出作用域后可写 public、可读 private, 原授权器仍在计数");

        // Removing the base authorizer leaves no authorizer at all.
        conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)?;
        conn.execute_batch("PRAGMA writable_schema = 1")?;
        println!("恢复: 移除基础授权器后 PRAGMA writable_schema 可用");
        Ok(())
    }

    #[test]
    fn nested_scopes_take_restriction_intersection() -> Result<()> {
        let conn = setup()?;
        conn.execute("INSERT INTO public (id, v) VALUES (2, 'b')", [])?;

        let outer =
            conn.authorizer_scope(ReadOnlyWhitelist::new().allow_tables(["public", "semi"]))?;
        let _: i64 = conn.query_row("SELECT count(*) FROM public", [], |r| r.get(0))?;
        let _: i64 = conn.query_row("SELECT count(*) FROM semi", [], |r| r.get(0))?;
        println!("允许: 外层作用域可读 public 与 semi");

        {
            let _inner = outer.nested(ReadOnlyWhitelist::new().allow_table("semi"))?;
            // Intersection of {public, semi} and {semi} is {semi}.
            let _: i64 = conn.query_row("SELECT count(*) FROM semi", [], |r| r.get(0))?;
            assert!(
                conn.query_row("SELECT count(*) FROM public", [], |r| r.get::<_, i64>(0))
                    .is_err()
            );
            println!("拒绝: 内层作用域交集只允许 semi, public 被拒");
        }

        // Leaving the inner scope restores the outer policy.
        let _: i64 = conn.query_row("SELECT count(*) FROM public", [], |r| r.get(0))?;
        println!("恢复: 退出内层后外层策略恢复, public 可读");

        drop(outer);
        conn.execute("INSERT INTO public (id, v) VALUES (3, 'c')", [])?;
        println!("恢复: 退出全部作用域后恢复可写");
        Ok(())
    }

    #[test]
    fn panic_unwind_restores_authorizer() -> Result<()> {
        let mut conn = setup()?;
        let calls = Arc::new(AtomicUsize::new(0));
        conn.authorizer(Some(counting_authorizer(Arc::clone(&calls))))?;

        let conn_ref = &conn;
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = conn_ref
                .authorizer_scope(ReadOnlyWhitelist::new().allow_table("public"))
                .unwrap();
            assert!(
                conn_ref
                    .execute("INSERT INTO public (id, v) VALUES (9, 'x')", [])
                    .is_err()
            );
            println!("拒绝: panic 前作用域内写 public");
            panic!("simulated failure inside authorizer scope");
        }))
        .is_err();
        assert!(panicked);

        // The guard was dropped during unwinding: the previous authorizer
        // (and its permissions) must be back.
        conn.execute("INSERT INTO public (id, v) VALUES (9, 'x')", [])?;
        let restored = calls.load(Ordering::SeqCst);
        let _: i64 = conn.query_row("SELECT count(*) FROM private", [], |r| r.get(0))?;
        assert!(calls.load(Ordering::SeqCst) > restored);
        println!("恢复: panic 展开后原授权器恢复且仍在计数");

        // And a subsequent normal enter/exit cycle restores again.
        {
            let _scope = conn.authorizer_scope(ReadOnlyWhitelist::new().allow_table("public"))?;
            assert!(conn.execute("DELETE FROM public", []).is_err());
            println!("拒绝: panic 后再次进入作用域仍生效");
        }
        conn.execute("DELETE FROM public WHERE id = 9", [])?;
        println!("恢复: 再次退出作用域后恢复可写");
        Ok(())
    }

    #[test]
    fn view_and_trigger_access_checked_by_actual_table() -> Result<()> {
        let conn = setup()?;
        // Custom closure policy: allow everything except touching private.
        let _scope = conn.authorizer_scope(|ctx: AuthContext<'_>| match ctx.action {
            AuthAction::Read { table_name, .. }
            | AuthAction::Insert { table_name }
            | AuthAction::Update { table_name, .. }
            | AuthAction::Delete { table_name }
                if table_name.eq_ignore_ascii_case("private") =>
            {
                Authorization::Deny
            }
            _ => Authorization::Allow,
        })?;

        // Direct read on private is denied.
        assert!(
            conn.query_row("SELECT secret FROM private", [], |r| r.get::<_, String>(0))
                .is_err()
        );
        // The same read through a view hits the real table and is denied too.
        assert!(
            conn.query_row("SELECT secret FROM v_private", [], |r| r
                .get::<_, String>(0))
                .is_err()
        );
        println!("拒绝: 直接及经 view 访问 private");

        // The trigger trg_leak writes to private when public is inserted into:
        // the actual write is reported and denied.
        assert!(
            conn.execute("INSERT INTO public (id, v) VALUES (5, 'e')", [])
                .is_err()
        );
        println!("拒绝: trigger 触发对 private 的实际写入");
        Ok(())
    }

    #[cfg(feature = "cache")]
    #[test]
    fn cached_statements_cannot_bypass_scope() -> Result<()> {
        let conn = setup()?;

        // Warm the cache with a statement reading private, then keep another
        // one checked out (alive) across the policy switch.
        {
            let mut stmt = conn.prepare_cached("SELECT secret FROM private")?;
            let s: String = stmt.query_row([], |r| r.get(0))?;
            assert_eq!(s, "top-secret");
        }
        let mut live = conn.prepare_cached("SELECT id FROM private")?;
        let _: i64 = live.query_row([], |r| r.get(0))?;
        println!("允许: 作用域外预缓存语句可执行");

        {
            let _scope = conn.authorizer_scope(ReadOnlyWhitelist::new().allow_table("public"))?;
            // The cache was flushed on entry: the same SQL is re-prepared and
            // now rejected by the authorizer.
            assert!(conn.prepare_cached("SELECT secret FROM private").is_err());
            println!("拒绝: 作用域内缓存失效, 重备 private 查询被拒");

            // A statement that was alive during the switch is discarded when
            // returned to the cache instead of being replayable later.
            drop(live);
            assert!(conn.prepare_cached("SELECT id FROM private").is_err());
            println!("拒绝: 切换时仍存活的语句归还缓存时被丢弃");
        }

        // After the scope, the cache was flushed again and the statement is
        // re-prepared under the restored (absent) authorizer.
        let mut stmt = conn.prepare_cached("SELECT secret FROM private")?;
        let s: String = stmt.query_row([], |r| r.get(0))?;
        assert_eq!(s, "top-secret");
        println!("恢复: 退出作用域后预缓存语句重新可用");
        Ok(())
    }

    #[cfg(feature = "functions")]
    #[test]
    fn unauthorized_functions_denied() -> Result<()> {
        use crate::functions::FunctionFlags;

        let conn = setup()?;
        conn.create_scalar_function("secret_fn", 0, FunctionFlags::SQLITE_DETERMINISTIC, |_| {
            Ok(42i64)
        })?;
        let n: i64 = conn.query_row("SELECT secret_fn()", [], |r| r.get(0))?;
        assert_eq!(n, 42);
        println!("允许: 作用域外调用自定义函数");

        {
            let _scope = conn.authorizer_scope(ReadOnlyWhitelist::new().allow_table("public"))?;
            // Built-in read-only functions are fine...
            let _: i64 = conn.query_row("SELECT count(*) FROM public", [], |r| r.get(0))?;
            // ...but an unauthorized function is rejected.
            assert!(
                conn.query_row("SELECT secret_fn()", [], |r| r.get::<_, i64>(0))
                    .is_err()
            );
            println!("拒绝: 作用域内未授权函数 secret_fn");
        }

        let n: i64 = conn.query_row("SELECT secret_fn()", [], |r| r.get(0))?;
        assert_eq!(n, 42);
        println!("恢复: 退出作用域后自定义函数可用");
        Ok(())
    }
}
