//! Transaction-scoped change collection.
//!
//! [`ChangeCollector`] wraps a [`Connection`] and records row changes
//! (database name, table name, action and rowid, in the order they actually
//! happen) reported through SQLite's update hook. Recorded events are handed
//! to the caller's callback only when the outermost managed transaction
//! commits successfully; events produced by rolled-back (sub-)transactions
//! are discarded and never leak into later deliveries.
//!
//! Transactions and savepoints must be managed through the collector's own
//! entry points ([`ChangeCollector::transaction`],
//! [`CollectorTransaction::savepoint`], ...). Raw SQL transaction control
//! statements (`BEGIN` / `COMMIT` / `END` / `ROLLBACK` / `SAVEPOINT` /
//! `RELEASE`) are rejected before execution by every statement entry point
//! exposed here, so the collector always observes exact transaction
//! boundaries.
//!
//! The collector deliberately does not try to reconstruct whole row images
//! or to merge the recorded changes into a final diff: events are delivered
//! exactly as they happened, in their original order.

use std::cell::RefCell;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::ffi;
use crate::hooks::Action;
use crate::pragma::Sql;
use crate::transaction::{DropBehavior, TransactionBehavior};
use crate::{Connection, Error, Params, Result, Row, Statement};

/// A single recorded row change.
///
/// Events are reported in the order the operations actually happened. No
/// attempt is made to reconstruct the row contents or to merge consecutive
/// events into a final diff.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChangeEvent {
    /// Name of the database that was changed ("main", "temp", ...).
    pub database_name: String,
    /// Name of the table that was changed.
    pub table_name: String,
    /// The kind of change (`SQLITE_INSERT`, `SQLITE_UPDATE` or
    /// `SQLITE_DELETE`).
    pub action: Action,
    /// The ROWID of the row that was changed.
    pub row_id: i64,
}

/// Callback invoked with the recorded events once the outermost managed
/// transaction has committed successfully.
///
/// The second argument is the underlying connection, so the callback may
/// safely run (re-entrant) queries against the just-committed state.
pub type ChangeCallback = dyn FnMut(&[ChangeEvent], &Connection);

enum Entry {
    Change(ChangeEvent),
    /// Boundary marker pushed when a managed savepoint starts.
    Mark,
}

#[derive(Default)]
struct Shared {
    /// Number of currently open managed transactions / savepoints.
    depth: usize,
    /// Recorded events and savepoint markers, in actual operation order.
    entries: Vec<Entry>,
}

impl Shared {
    fn lock(shared: &Arc<Mutex<Self>>) -> MutexGuard<'_, Self> {
        shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn push_mark(&mut self) {
        self.entries.push(Entry::Mark);
        self.depth += 1;
    }

    /// Discard events recorded since the most recent savepoint marker; the
    /// marker itself is kept because the savepoint remains active.
    fn rollback_to_mark(&mut self) {
        while matches!(self.entries.last(), Some(Entry::Change(_))) {
            self.entries.pop();
        }
    }

    /// Merge the most recent savepoint's events into its parent.
    fn release_mark(&mut self) {
        if let Some(pos) = self.entries.iter().rposition(|e| matches!(e, Entry::Mark)) {
            self.entries.remove(pos);
        }
        self.depth = self.depth.saturating_sub(1);
    }

    /// The outermost transaction ended: reset bookkeeping and return the
    /// surviving events in their original order.
    fn take_events(&mut self) -> Vec<ChangeEvent> {
        self.depth = 0;
        std::mem::take(&mut self.entries)
            .into_iter()
            .filter_map(|e| match e {
                Entry::Change(c) => Some(c),
                Entry::Mark => None,
            })
            .collect()
    }

    /// The outermost transaction was rolled back (or its commit failed):
    /// drop everything that was recorded.
    fn discard(&mut self) {
        self.depth = 0;
        self.entries.clear();
    }
}

/// A connection wrapper that collects row changes per transaction and
/// delivers them to a callback only when the outermost transaction commits
/// successfully.
///
/// See the [module-level documentation](self) for details.
pub struct ChangeCollector {
    conn: Connection,
    shared: Arc<Mutex<Shared>>,
    on_commit: RefCell<Box<ChangeCallback>>,
}

impl ChangeCollector {
    /// Wrap `conn`, registering an internal update hook, and invoke
    /// `on_commit` with the recorded events each time the outermost managed
    /// transaction commits successfully.
    ///
    /// The callback receives the underlying connection as its second
    /// argument and may safely run re-entrant queries against it.
    pub fn new<F>(mut conn: Connection, on_commit: F) -> Result<Self>
    where
        F: FnMut(&[ChangeEvent], &Connection) + 'static,
    {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let hook_shared = Arc::clone(&shared);
        conn.update_hook(Some(move |action, db: &str, tbl: &str, row_id| {
            let mut shared = Shared::lock(&hook_shared);
            // Only changes made inside a managed transaction are recorded.
            if shared.depth > 0 {
                shared.entries.push(Entry::Change(ChangeEvent {
                    database_name: db.to_owned(),
                    table_name: tbl.to_owned(),
                    action,
                    row_id,
                }));
            }
        }))?;
        Ok(Self {
            conn,
            shared,
            on_commit: RefCell::new(Box::new(on_commit)),
        })
    }

    /// Begin a new managed transaction with the default behavior (DEFERRED).
    ///
    /// The transaction defaults to rolling back when it is dropped. Use
    /// [`CollectorTransaction::commit`] to deliver the recorded events.
    ///
    /// # Failure
    ///
    /// Will return `Err` if a managed transaction is already open or if the
    /// underlying SQLite call fails.
    #[inline]
    pub fn transaction(&mut self) -> Result<CollectorTransaction<'_>> {
        self.transaction_with_behavior(TransactionBehavior::Deferred)
    }

    /// Begin a new managed transaction with the specified behavior.
    ///
    /// See [`transaction`](ChangeCollector::transaction).
    #[inline]
    pub fn transaction_with_behavior(
        &mut self,
        behavior: TransactionBehavior,
    ) -> Result<CollectorTransaction<'_>> {
        if Shared::lock(&self.shared).depth != 0 {
            return Err(Error::SqliteFailure(
                ffi::Error::new(ffi::SQLITE_MISUSE),
                Some("a managed transaction is already open".to_owned()),
            ));
        }
        let query = match behavior {
            TransactionBehavior::Deferred => "BEGIN DEFERRED",
            TransactionBehavior::Immediate => "BEGIN IMMEDIATE",
            TransactionBehavior::Exclusive => "BEGIN EXCLUSIVE",
        };
        self.conn.execute_batch(query)?;
        Shared::lock(&self.shared).depth = 1;
        Ok(CollectorTransaction {
            guard: GuardedConnection::new(self),
            drop_behavior: DropBehavior::Rollback,
            finished: false,
        })
    }

    /// Execute a batch of SQL statements outside of a managed transaction.
    ///
    /// Raw transaction control statements are rejected before execution.
    #[inline]
    pub fn execute_batch(&self, sql: &str) -> Result<()> {
        self.guard().execute_batch(sql)
    }

    /// Prepare and execute a single SQL statement outside of a managed
    /// transaction.
    ///
    /// Raw transaction control statements are rejected before execution.
    #[inline]
    pub fn execute<P: Params>(&self, sql: &str, params: P) -> Result<usize> {
        self.guard().execute(sql, params)
    }

    /// Prepare a SQL statement for execution outside of a managed
    /// transaction.
    ///
    /// Raw transaction control statements are rejected before execution.
    #[inline]
    pub fn prepare(&self, sql: &str) -> Result<Statement<'_>> {
        self.guard().prepare(sql)
    }

    /// Run a query that is expected to return a single row, outside of a
    /// managed transaction.
    #[inline]
    pub fn query_row<T, P, F>(&self, sql: &str, params: P, f: F) -> Result<T>
    where
        P: Params,
        F: FnOnce(&Row<'_>) -> Result<T>,
    {
        self.guard().query_row(sql, params, f)
    }

    #[inline]
    fn guard(&self) -> GuardedConnection<'_> {
        GuardedConnection::new(self)
    }

    /// Called when the outermost transaction committed successfully.
    fn deliver(&self, events: &[ChangeEvent]) {
        if events.is_empty() {
            return;
        }
        // No lock on `shared` is held here, so the callback may freely run
        // re-entrant queries against the connection.
        let mut callback = self.on_commit.borrow_mut();
        callback(events, &self.conn);
    }
}

/// A restricted view of the collector's connection.
///
/// All statement entry points reject raw SQL transaction control statements
/// (`BEGIN` / `COMMIT` / `END` / `ROLLBACK` / `SAVEPOINT` / `RELEASE`)
/// before execution; transactions and savepoints must be managed through
/// [`ChangeCollector`], [`CollectorTransaction`] and [`CollectorSavepoint`].
#[derive(Clone, Copy)]
pub struct GuardedConnection<'c> {
    collector: &'c ChangeCollector,
}

impl<'c> GuardedConnection<'c> {
    #[inline]
    fn new(collector: &'c ChangeCollector) -> Self {
        Self { collector }
    }

    /// Execute a batch of SQL statements, rejecting raw transaction control
    /// statements before execution.
    #[inline]
    pub fn execute_batch(&self, sql: &str) -> Result<()> {
        reject_txn_control(sql)?;
        self.collector.conn.execute_batch(sql)
    }

    /// Prepare and execute a single SQL statement, rejecting raw
    /// transaction control statements before execution.
    #[inline]
    pub fn execute<P: Params>(&self, sql: &str, params: P) -> Result<usize> {
        reject_txn_control(sql)?;
        self.collector.conn.execute(sql, params)
    }

    /// Prepare a SQL statement, rejecting raw transaction control
    /// statements before execution.
    #[inline]
    pub fn prepare(&self, sql: &str) -> Result<Statement<'c>> {
        reject_txn_control(sql)?;
        self.collector.conn.prepare(sql)
    }

    /// Run a query that is expected to return a single row, rejecting raw
    /// transaction control statements before execution.
    #[inline]
    pub fn query_row<T, P, F>(&self, sql: &str, params: P, f: F) -> Result<T>
    where
        P: Params,
        F: FnOnce(&Row<'_>) -> Result<T>,
    {
        reject_txn_control(sql)?;
        self.collector.conn.query_row(sql, params, f)
    }

    /// Check whether the connection is in autocommit mode.
    #[inline]
    pub fn is_autocommit(&self) -> bool {
        self.collector.conn.is_autocommit()
    }
}

/// A managed transaction on a [`ChangeCollector`].
///
/// Like [`crate::Transaction`], managed transactions roll back by default
/// when dropped. Recorded change events are delivered to the collector's
/// callback only when the outermost transaction commits successfully.
pub struct CollectorTransaction<'c> {
    guard: GuardedConnection<'c>,
    drop_behavior: DropBehavior,
    finished: bool,
}

impl CollectorTransaction<'_> {
    /// Start a new managed savepoint, allowing nested transactions.
    #[inline]
    pub fn savepoint(&mut self) -> Result<CollectorSavepoint<'_>> {
        CollectorSavepoint::with_name_(self.guard, "_rusqlite_sp")
    }

    /// Start a new managed savepoint with a user-provided name.
    #[inline]
    pub fn savepoint_with_name<T: Into<String>>(
        &mut self,
        name: T,
    ) -> Result<CollectorSavepoint<'_>> {
        CollectorSavepoint::with_name_(self.guard, name)
    }

    /// Get the current setting for what happens to the transaction when it
    /// is dropped.
    #[inline]
    #[must_use]
    pub fn drop_behavior(&self) -> DropBehavior {
        self.drop_behavior
    }

    /// Configure the transaction to perform the specified action when it is
    /// dropped.
    #[inline]
    pub fn set_drop_behavior(&mut self, drop_behavior: DropBehavior) {
        self.drop_behavior = drop_behavior;
    }

    /// A convenience method which consumes and commits the transaction.
    ///
    /// On success, the recorded change events are delivered to the
    /// collector's callback. On failure (e.g. a deferred constraint
    /// violation reported by `COMMIT`), the transaction is rolled back and
    /// the recorded events are discarded.
    #[inline]
    pub fn commit(mut self) -> Result<()> {
        self.commit_()
    }

    fn commit_(&mut self) -> Result<()> {
        self.finished = true;
        match self.guard.collector.conn.execute_batch("COMMIT") {
            Ok(()) => {
                let events = Shared::lock(&self.guard.collector.shared).take_events();
                self.guard.collector.deliver(&events);
                Ok(())
            }
            Err(e) => {
                // The commit failed (e.g. a deferred constraint violation);
                // the transaction may still be active, so roll it back and
                // make sure no recorded event can leak into a later
                // delivery.
                let _ = self.guard.collector.conn.execute_batch("ROLLBACK");
                Shared::lock(&self.guard.collector.shared).discard();
                Err(e)
            }
        }
    }

    /// A convenience method which consumes and rolls back the transaction,
    /// discarding all recorded events.
    #[inline]
    pub fn rollback(mut self) -> Result<()> {
        self.rollback_()
    }

    fn rollback_(&mut self) -> Result<()> {
        self.finished = true;
        let r = self.guard.collector.conn.execute_batch("ROLLBACK");
        Shared::lock(&self.guard.collector.shared).discard();
        r
    }

    /// Consumes the transaction, committing or rolling back according to
    /// the current setting (see `drop_behavior`).
    #[inline]
    pub fn finish(mut self) -> Result<()> {
        self.finish_()
    }

    fn finish_(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        match self.drop_behavior() {
            DropBehavior::Commit => self.commit_().or_else(|_| self.rollback_()),
            DropBehavior::Rollback => self.rollback_(),
            DropBehavior::Ignore => Ok(()),
            DropBehavior::Panic => panic!("Transaction dropped unexpectedly."),
        }
    }
}

impl<'c> Deref for CollectorTransaction<'c> {
    type Target = GuardedConnection<'c>;

    #[inline]
    fn deref(&self) -> &GuardedConnection<'c> {
        &self.guard
    }
}

#[expect(unused_must_use)]
impl Drop for CollectorTransaction<'_> {
    #[inline]
    fn drop(&mut self) {
        self.finish_();
    }
}

/// A managed savepoint on a [`ChangeCollector`].
///
/// Like [`crate::Savepoint`], managed savepoints roll back by default when
/// dropped. Rolling back a savepoint discards only the events recorded
/// since that savepoint started; releasing it merges them into the parent.
pub struct CollectorSavepoint<'c> {
    guard: GuardedConnection<'c>,
    name: String,
    drop_behavior: DropBehavior,
    committed: bool,
}

impl<'c> CollectorSavepoint<'c> {
    fn with_name_<T: Into<String>>(guard: GuardedConnection<'c>, name: T) -> Result<Self> {
        let name = name.into();
        let sql = cmd("SAVEPOINT", false, name.as_str())?;
        guard.collector.conn.execute_batch(sql.as_str())?;
        Shared::lock(&guard.collector.shared).push_mark();
        Ok(Self {
            guard,
            name,
            drop_behavior: DropBehavior::Rollback,
            committed: false,
        })
    }

    /// Begin a nested managed savepoint.
    #[inline]
    pub fn savepoint(&mut self) -> Result<CollectorSavepoint<'_>> {
        CollectorSavepoint::with_name_(self.guard, "_rusqlite_sp")
    }

    /// Begin a nested managed savepoint with a user-provided name.
    #[inline]
    pub fn savepoint_with_name<T: Into<String>>(
        &mut self,
        name: T,
    ) -> Result<CollectorSavepoint<'_>> {
        CollectorSavepoint::with_name_(self.guard, name)
    }

    /// Get the current setting for what happens to the savepoint when it is
    /// dropped.
    #[inline]
    #[must_use]
    pub fn drop_behavior(&self) -> DropBehavior {
        self.drop_behavior
    }

    /// Configure the savepoint to perform the specified action when it is
    /// dropped.
    #[inline]
    pub fn set_drop_behavior(&mut self, drop_behavior: DropBehavior) {
        self.drop_behavior = drop_behavior;
    }

    /// A convenience method which consumes and releases the savepoint,
    /// merging the events recorded since it started into its parent.
    #[inline]
    pub fn commit(mut self) -> Result<()> {
        self.commit_()
    }

    fn commit_(&mut self) -> Result<()> {
        let sql = cmd("RELEASE", false, self.name.as_str())?;
        self.guard.collector.conn.execute_batch(sql.as_str())?;
        Shared::lock(&self.guard.collector.shared).release_mark();
        self.committed = true;
        Ok(())
    }

    /// Rolls back to this savepoint, discarding the events recorded since
    /// it started.
    ///
    /// ## Note
    ///
    /// Unlike transactions, savepoints remain active after they have been
    /// rolled back, and can be rolled back again or released.
    #[inline]
    pub fn rollback(&mut self) -> Result<()> {
        let sql = cmd("ROLLBACK", true, self.name.as_str())?;
        self.guard.collector.conn.execute_batch(sql.as_str())?;
        Shared::lock(&self.guard.collector.shared).rollback_to_mark();
        Ok(())
    }

    /// Consumes the savepoint, releasing or rolling back according to the
    /// current setting (see `drop_behavior`).
    #[inline]
    pub fn finish(mut self) -> Result<()> {
        self.finish_()
    }

    fn finish_(&mut self) -> Result<()> {
        if self.committed {
            return Ok(());
        }
        match self.drop_behavior() {
            DropBehavior::Commit => self
                .commit_()
                .or_else(|_| self.rollback().and_then(|()| self.commit_())),
            DropBehavior::Rollback => self.rollback().and_then(|()| self.commit_()),
            DropBehavior::Ignore => Ok(()),
            DropBehavior::Panic => panic!("Savepoint dropped unexpectedly."),
        }
    }
}

impl<'c> Deref for CollectorSavepoint<'c> {
    type Target = GuardedConnection<'c>;

    #[inline]
    fn deref(&self) -> &GuardedConnection<'c> {
        &self.guard
    }
}

#[expect(unused_must_use)]
impl Drop for CollectorSavepoint<'_> {
    #[inline]
    fn drop(&mut self) {
        self.finish_();
    }
}

fn cmd(cmd: &'static str, to: bool, name: &str) -> Result<Sql> {
    let mut sql = Sql::new();
    sql.push_keyword(cmd)?;
    sql.push_space();
    if to {
        sql.push_keyword("TO")?;
        sql.push_space();
    }
    sql.push_identifier(name);
    Ok(sql)
}

/// Reject raw SQL transaction control statements before execution.
///
/// Transactions and savepoints must go through the collector's managed
/// entry points so that event bookkeeping always matches the actual
/// transaction boundaries.
fn reject_txn_control(sql: &str) -> Result<()> {
    let sanitized = sanitize_sql(sql);
    for stmt in sanitized.split(';') {
        let token: String = stmt
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        match token.to_ascii_uppercase().as_str() {
            "BEGIN" | "COMMIT" | "END" | "ROLLBACK" | "SAVEPOINT" | "RELEASE" | "TRANSACTION" => {
                return Err(Error::SqlInputError {
                    error: ffi::Error::new(ffi::SQLITE_ERROR),
                    msg: "raw transaction control statements are not allowed; use the \
                          ChangeCollector transaction/savepoint entry points"
                        .to_owned(),
                    sql: sql.to_owned(),
                    offset: 0,
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Blank out string literals, quoted identifiers and comments so that
/// transaction control keywords hidden inside them are not mistaken for
/// real statements.
fn sanitize_sql(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            q @ ('\'' | '"' | '`') => {
                out.push(' ');
                loop {
                    match chars.next() {
                        None => break,
                        Some(n) if n == q => {
                            // A doubled quote is an escaped quote, not the end.
                            if chars.peek() == Some(&q) {
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        Some(_) => {}
                    }
                }
            }
            '[' => {
                out.push(' ');
                for n in chars.by_ref() {
                    if n == ']' {
                        break;
                    }
                }
            }
            '-' if chars.peek() == Some(&'-') => {
                chars.next();
                for n in chars.by_ref() {
                    if n == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = '\0';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
            }
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Connection;
    use std::rc::Rc;

    type Delivered = Rc<RefCell<Vec<Vec<ChangeEvent>>>>;

    fn new_collector() -> Result<(ChangeCollector, Delivered)> {
        let conn = Connection::open_in_memory()?;
        let delivered: Delivered = Rc::new(RefCell::new(Vec::new()));
        let d = Rc::clone(&delivered);
        let collector = ChangeCollector::new(conn, move |events, _conn| {
            d.borrow_mut().push(events.to_vec());
        })?;
        Ok((collector, delivered))
    }

    fn insert_event(table: &str, row_id: i64) -> ChangeEvent {
        ChangeEvent {
            database_name: "main".to_owned(),
            table_name: table.to_owned(),
            action: Action::SQLITE_INSERT,
            row_id,
        }
    }

    #[test]
    fn test_commit_delivers_only_surviving_changes() -> Result<()> {
        let (mut c, delivered) = new_collector()?;
        c.execute_batch("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")?;
        {
            let mut tx = c.transaction()?;
            tx.execute("INSERT INTO t(v) VALUES ('a')", [])?;
            {
                let mut sp = tx.savepoint()?;
                sp.execute("UPDATE t SET v = 'b' WHERE id = 1", [])?;
                sp.rollback()?;
            }
            tx.commit()?;
        }
        let d = delivered.borrow();
        assert_eq!(d.len(), 1, "expected a single delivery, got {d:?}");
        assert_eq!(d[0], vec![insert_event("t", 1)]);
        Ok(())
    }

    #[test]
    fn test_nested_savepoint_release() -> Result<()> {
        let (mut c, delivered) = new_collector()?;
        c.execute_batch("CREATE TABLE t(id INTEGER PRIMARY KEY)")?;
        {
            let mut tx = c.transaction()?;
            tx.execute("INSERT INTO t VALUES (1)", [])?;
            {
                let mut sp1 = tx.savepoint()?;
                sp1.execute("INSERT INTO t VALUES (2)", [])?;
                {
                    let sp2 = sp1.savepoint()?;
                    sp2.execute("INSERT INTO t VALUES (3)", [])?;
                    sp2.commit()?;
                }
                sp1.commit()?;
            }
            tx.commit()?;
        }
        let d = delivered.borrow();
        assert_eq!(d.len(), 1);
        assert_eq!(
            d[0],
            vec![
                insert_event("t", 1),
                insert_event("t", 2),
                insert_event("t", 3)
            ]
        );
        Ok(())
    }

    #[test]
    fn test_outermost_rollback_delivers_nothing() -> Result<()> {
        let (mut c, delivered) = new_collector()?;
        c.execute_batch("CREATE TABLE t(id INTEGER PRIMARY KEY)")?;
        {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO t VALUES (1)", [])?;
            tx.rollback()?;
        }
        {
            // dropped without commit: rolled back by default
            let tx = c.transaction()?;
            tx.execute("INSERT INTO t VALUES (5)", [])?;
        }
        assert!(delivered.borrow().is_empty());
        {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO t VALUES (2)", [])?;
            tx.commit()?;
        }
        let d = delivered.borrow();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0], vec![insert_event("t", 2)]);
        Ok(())
    }

    #[test]
    fn test_failed_commit_delivers_nothing() -> Result<()> {
        let (mut c, delivered) = new_collector()?;
        c.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE parent(id INTEGER PRIMARY KEY);
             CREATE TABLE child(
                 id INTEGER PRIMARY KEY,
                 pid INTEGER NOT NULL REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED
             );",
        )?;
        {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO child VALUES (1, 999)", [])?;
            assert!(
                tx.commit().is_err(),
                "deferred FK violation must fail COMMIT"
            );
        }
        assert!(delivered.borrow().is_empty());
        // The collector is usable again and nothing leaked from the failed
        // transaction.
        {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO parent VALUES (1)", [])?;
            tx.execute("INSERT INTO child VALUES (1, 1)", [])?;
            tx.commit()?;
        }
        let d = delivered.borrow();
        assert_eq!(d.len(), 1);
        assert_eq!(
            d[0],
            vec![insert_event("parent", 1), insert_event("child", 1)]
        );
        Ok(())
    }

    #[test]
    fn test_raw_txn_control_rejected() -> Result<()> {
        let (mut c, delivered) = new_collector()?;
        c.execute_batch("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")?;
        assert!(c.execute_batch("BEGIN").is_err());
        assert!(c.execute_batch("COMMIT").is_err());
        assert!(c.execute_batch(" -- lead comment\nROLLBACK").is_err());
        assert!(c.execute_batch("/* block */ SAVEPOINT s").is_err());
        {
            let tx = c.transaction()?;
            assert!(tx.execute_batch("COMMIT").is_err());
            assert!(tx.execute_batch("END TRANSACTION").is_err());
            assert!(tx.execute_batch("SAVEPOINT s").is_err());
            assert!(tx.execute_batch("RELEASE s").is_err());
            assert!(tx.execute_batch("ROLLBACK TO s").is_err());
            assert!(
                tx.execute("INSERT INTO t(v) VALUES ('x'); COMMIT", [])
                    .is_err()
            );
            // A string literal that merely looks like transaction control is
            // not a transaction control statement.
            tx.execute("INSERT INTO t(v) VALUES ('BEGIN; COMMIT;')", [])?;
            tx.commit()?;
        }
        let d = delivered.borrow();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0], vec![insert_event("t", 1)]);
        Ok(())
    }

    #[test]
    fn test_callback_reentrant_query() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        let seen: Rc<RefCell<Vec<(usize, i64)>>> = Rc::new(RefCell::new(Vec::new()));
        let s = Rc::clone(&seen);
        let mut c = ChangeCollector::new(conn, move |events, conn| {
            // Re-entrant query against the just-committed state.
            let n: i64 = conn
                .query_row("SELECT count(*) FROM t", [], |r| r.get(0))
                .unwrap();
            s.borrow_mut().push((events.len(), n));
        })?;
        c.execute_batch("CREATE TABLE t(id INTEGER PRIMARY KEY)")?;
        {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO t VALUES (1)", [])?;
            tx.execute("INSERT INTO t VALUES (2)", [])?;
            tx.commit()?;
        }
        assert_eq!(*seen.borrow(), vec![(2usize, 2i64)]);
        Ok(())
    }

    #[test]
    fn test_event_order_and_actions() -> Result<()> {
        let (mut c, delivered) = new_collector()?;
        c.execute_batch("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")?;
        {
            let tx = c.transaction()?;
            tx.execute("INSERT INTO t(v) VALUES ('a')", [])?;
            tx.execute("UPDATE t SET v = 'b' WHERE id = 1", [])?;
            tx.execute("DELETE FROM t WHERE id = 1", [])?;
            tx.commit()?;
        }
        let d = delivered.borrow();
        assert_eq!(d.len(), 1);
        let actions: Vec<Action> = d[0].iter().map(|e| e.action).collect();
        assert_eq!(
            actions,
            vec![
                Action::SQLITE_INSERT,
                Action::SQLITE_UPDATE,
                Action::SQLITE_DELETE
            ]
        );
        assert!(d[0].iter().all(|e| e.row_id == 1 && e.table_name == "t"));
        Ok(())
    }
}
