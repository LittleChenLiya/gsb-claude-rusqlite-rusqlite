//! Transaction change collection.
//!
//! This module provides [`TransactionChangeCollector`], which records the row
//! changes observed through the [update hook](crate::hooks::update_hook) while a
//! managed transaction or savepoint is active. The recorded changes are handed to
//! a caller-supplied callback only when the outermost transaction commits
//! successfully; changes made inside a savepoint that is rolled back are
//! discarded, and a failed commit never leaks events.
//!
//! Only transactions and savepoints created through the collector's entry points
//! ([`TransactionChangeCollector::transaction`] /
//! [`TransactionChangeCollector::savepoint`]) are tracked. Raw SQL transaction
//! control statements (`BEGIN`, `COMMIT`, `ROLLBACK`, `SAVEPOINT`, `RELEASE`,
//! `ROLLBACK TO`) are rejected by an authorizer installed by the collector while
//! it is alive.
//!
//! The collector does not attempt to reconstruct whole rows or merge changes into
//! a final diff; each observed insert, update or delete is delivered in the order
//! in which it actually occurred.

use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::hooks::{Action, AuthAction, AuthContext, Authorization};
use crate::transaction::{cmd, DropBehavior, TransactionBehavior};
use crate::{Connection, Result};

/// A single row change observed within a managed transaction.
///
/// The collector records the database name, table name, operation and rowid for
/// every insert, update or delete performed while a managed transaction is
/// active, in the order in which they occurred.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransactionChange {
    /// The operation that was performed (`SQLITE_INSERT`, `SQLITE_UPDATE` or
    /// `SQLITE_DELETE`).
    pub action: Action,
    /// The name of the database the change was made in (e.g. `"main"`,
    /// `"temp"`).
    pub database: String,
    /// The name of the table the change was made in.
    pub table: String,
    /// The [rowid](https://www.sqlite.org/lang_createtable.html#rowid) of the
    /// changed row.
    pub rowid: i64,
}

/// The mutable state shared between the hooks and the managed transaction
/// wrappers.
struct CollectorState {
    /// Changes recorded while a managed transaction is active.
    events: Vec<TransactionChange>,
    /// Nesting depth of managed transactions/savepoints. `0` means autocommit.
    depth: usize,
    /// Active savepoints: `(name, events_len_at_savepoint)`.
    savepoints: Vec<(String, usize)>,
    /// When `true`, the authorizer allows transaction-control statements. This
    /// is set only while a managed transaction/savepoint operation is executing.
    allow_managed: bool,
    /// Counter used to generate unique savepoint names.
    savepoint_counter: usize,
}

impl CollectorState {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            depth: 0,
            savepoints: Vec::new(),
            allow_managed: false,
            savepoint_counter: 0,
        }
    }

    fn next_savepoint_name(&mut self) -> String {
        let n = self.savepoint_counter;
        self.savepoint_counter += 1;
        format!("_rusqlite_sp_{n}")
    }

    /// Record a change. Changes are only retained while a managed transaction is
    /// active; statements executed in autocommit are ignored.
    fn record(&mut self, change: TransactionChange) {
        if self.depth > 0 {
            self.events.push(change);
        }
    }

    /// Record a savepoint checkpoint at the current event position.
    fn checkpoint(&mut self, name: String) {
        let len = self.events.len();
        self.savepoints.push((name, len));
    }

    /// Roll back to the named savepoint: discard every change recorded after the
    /// savepoint was taken. The savepoint itself remains active (matching SQLite
    /// semantics), so it is kept in the stack with its checkpoint.
    fn rollback_to(&mut self, name: &str) {
        if let Some(pos) = self.savepoints.iter().rposition(|(n, _)| n == name) {
            let (_, len) = self.savepoints[pos];
            self.events.truncate(len);
            // Keep the named savepoint; savepoints taken after it are removed.
            self.savepoints.truncate(pos + 1);
        }
    }

    /// Release the named savepoint. Changes recorded while it was active are
    /// retained; only the checkpoint metadata is removed.
    fn release(&mut self, name: &str) {
        if let Some(pos) = self.savepoints.iter().rposition(|(n, _)| n == name) {
            self.savepoints.truncate(pos);
        }
    }

    fn take_events(&mut self) -> Vec<TransactionChange> {
        std::mem::take(&mut self.events)
    }

    /// Discard all recorded changes and reset the transaction nesting state.
    fn clear(&mut self) {
        self.events.clear();
        self.savepoints.clear();
        self.depth = 0;
    }
}
/// A callback invoked with the connection and the recorded changes after the
/// outermost managed transaction commits successfully.
type ChangeCallback = Box<dyn FnMut(&Connection, &[TransactionChange]) + Send>;

/// Shared ownership of the collector state, held by the hooks and the managed
/// transaction wrappers.
struct Inner {
    state: Mutex<CollectorState>,
    callback: Mutex<Option<ChangeCallback>>,
}

impl Inner {
    /// Deliver the recorded changes to the caller. The callback is invoked with
    /// the connection so that it can reentrantly query the database now that the
    /// outermost transaction has committed successfully.
    fn deliver(&self, conn: &Connection) {
        let events = self.state.lock().unwrap().take_events();
        if events.is_empty() {
            return;
        }
        let mut cb = self.callback.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cb) = cb.as_mut() {
            cb(conn, &events);
        }
    }

    fn clear(&self) {
        self.state.lock().unwrap().clear();
    }

    /// Reset transaction nesting state after a successful outermost commit.
    /// The recorded events are left in place for [`deliver`](Inner::deliver) to
    /// take.
    fn commit_succeeded(&self) {
        let mut s = self.state.lock().unwrap();
        s.depth = 0;
        s.savepoints.clear();
    }
}

/// Guard that resets the `allow_managed` flag when a managed operation finishes,
/// even if it panics.
struct AllowGuard<'a>(&'a Inner);

impl Drop for AllowGuard<'_> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().allow_managed = false;
    }
}

/// Run a transaction/savepoint operation issued by a managed wrapper. The
/// authorizer only allows transaction-control statements while this is running.
fn run_managed<T>(inner: &Inner, f: impl FnOnce() -> Result<T>) -> Result<T> {
    inner.state.lock().unwrap().allow_managed = true;
    let _guard = AllowGuard(inner);
    f()
}

/// A collector of row changes observed within managed transactions and
/// savepoints.
///
/// The collector installs an [update hook](crate::hooks::update_hook) and an
/// [authorizer](crate::hooks::authorizer) on the connection. Use
/// [`TransactionChangeCollector::transaction`] and
/// [`TransactionChangeCollector::savepoint`] to begin managed transactions and
/// savepoints; changes made within them are recorded and delivered to the
/// callback only when the outermost transaction commits successfully.
///
/// Raw SQL transaction control statements (`BEGIN`, `COMMIT`, `ROLLBACK`,
/// `SAVEPOINT`, `RELEASE`, `ROLLBACK TO`) are rejected while the collector is
/// alive.
///
/// ## Example
///
/// ```rust,no_run
/// # use rusqlite::{Connection, Result};
/// # use rusqlite::hooks::txn_changes::{TransactionChange, TransactionChangeCollector};
/// fn main() -> Result<()> {
///     let mut db = Connection::open_in_memory()?;
///     db.execute_batch("CREATE TABLE foo (id INTEGER PRIMARY KEY, v TEXT)")?;
///
///     let collector = TransactionChangeCollector::new(
///         &mut db,
///         |_conn, changes: &[TransactionChange]| {
///             for change in changes {
///                 println!("{:?} {} {} {}", change.action, change.database, change.table, change.rowid);
///             }
///         },
///     )?;
///
///     let tx = collector.transaction(&mut db)?;
///     tx.execute_batch("INSERT INTO foo (v) VALUES ('a')")?;
///     tx.commit()?; // callback fires here with the single insert
///     Ok(())
/// }
/// ```
pub struct TransactionChangeCollector {
    inner: Arc<Inner>,
}

impl TransactionChangeCollector {
    /// Create a new collector and install its hooks on `conn`.
    ///
    /// `callback` is invoked with the connection and the recorded changes after
    /// the outermost managed transaction commits successfully. It is not invoked
    /// for transactions that are rolled back or that fail to commit.
    ///
    /// The collector installs an [update hook](crate::hooks::update_hook) and an
    /// [authorizer](crate::hooks::authorizer) on `conn`. The authorizer rejects
    /// raw SQL transaction control statements while the collector is alive; if
    /// you install another authorizer afterwards, the collector's protection is
    /// replaced and raw SQL transaction control will no longer be rejected.
    pub fn new<F>(conn: &mut Connection, callback: F) -> Result<Self>
    where
        F: FnMut(&Connection, &[TransactionChange]) + Send + 'static,
    {
        let inner = Arc::new(Inner {
            state: Mutex::new(CollectorState::new()),
            callback: Mutex::new(Some(Box::new(callback))),
        });
        install_update_hook(conn, &inner)?;
        install_authorizer(conn, &inner)?;
        Ok(TransactionChangeCollector { inner })
    }

    /// Begin a new managed transaction with the default (DEFERRED) behavior.
    pub fn transaction<'conn>(
        &self,
        conn: &'conn mut Connection,
    ) -> Result<ManagedTransaction<'conn>> {
        ManagedTransaction::begin(conn, self.inner.clone(), TransactionBehavior::Deferred)
    }

    /// Begin a new managed transaction with the specified behavior.
    pub fn transaction_with_behavior<'conn>(
        &self,
        conn: &'conn mut Connection,
        behavior: TransactionBehavior,
    ) -> Result<ManagedTransaction<'conn>> {
        ManagedTransaction::begin(conn, self.inner.clone(), behavior)
    }

    /// Begin a new managed savepoint.
    pub fn savepoint<'conn>(
        &self,
        conn: &'conn mut Connection,
    ) -> Result<ManagedSavepoint<'conn>> {
        ManagedSavepoint::begin(conn, self.inner.clone())
    }

    /// Begin a new managed savepoint with a user-provided name.
    pub fn savepoint_with_name<'conn, T: Into<String>>(
        &self,
        conn: &'conn mut Connection,
        name: T,
    ) -> Result<ManagedSavepoint<'conn>> {
        ManagedSavepoint::with_name(conn, self.inner.clone(), name.into())
    }
}

fn install_update_hook(conn: &mut Connection, inner: &Arc<Inner>) -> Result<()> {
    let inner = inner.clone();
    conn.update_hook(Some(
        move |action: Action, database: &str, table: &str, rowid: i64| {
            inner.state.lock().unwrap().record(TransactionChange {
                action,
                database: database.to_owned(),
                table: table.to_owned(),
                rowid,
            });
        },
    ))
}

fn install_authorizer(conn: &mut Connection, inner: &Arc<Inner>) -> Result<()> {
    let inner = inner.clone();
    conn.authorizer(Some(move |ctx: AuthContext<'_>| match ctx.action {
        AuthAction::Transaction { .. } | AuthAction::Savepoint { .. } => {
            if inner.state.lock().unwrap().allow_managed {
                Authorization::Allow
            } else {
                Authorization::Deny
            }
        }
        _ => Authorization::Allow,
    }))
}

/// A managed transaction, created by a [`TransactionChangeCollector`].
///
/// Changes made within a managed transaction are recorded and delivered to the
/// collector's callback only when the outermost transaction commits successfully.
///
/// ## Note
///
/// Transactions roll back by default. Use [`commit`](ManagedTransaction::commit)
/// to explicitly commit, or
/// [`set_drop_behavior`](ManagedTransaction::set_drop_behavior) to change what
/// happens when the transaction is dropped.
pub struct ManagedTransaction<'conn> {
    conn: &'conn Connection,
    inner: Arc<Inner>,
    drop_behavior: DropBehavior,
    finished: bool,
}

impl<'conn> ManagedTransaction<'conn> {
    fn begin(
        conn: &'conn mut Connection,
        inner: Arc<Inner>,
        behavior: TransactionBehavior,
    ) -> Result<Self> {
        let query = match behavior {
            TransactionBehavior::Deferred => "BEGIN DEFERRED",
            TransactionBehavior::Immediate => "BEGIN IMMEDIATE",
            TransactionBehavior::Exclusive => "BEGIN EXCLUSIVE",
        };
        run_managed(&inner, || conn.execute_batch(query))?;
        inner.state.lock().unwrap().depth += 1;
        Ok(ManagedTransaction {
            conn,
            inner,
            drop_behavior: DropBehavior::Rollback,
            finished: false,
        })
    }

    /// Begin a nested managed savepoint.
    pub fn savepoint(&mut self) -> Result<ManagedSavepoint<'_>> {
        ManagedSavepoint::begin(self.conn, self.inner.clone())
    }

    /// Begin a nested managed savepoint with a user-provided name.
    pub fn savepoint_with_name<T: Into<String>>(
        &mut self,
        name: T,
    ) -> Result<ManagedSavepoint<'_>> {
        ManagedSavepoint::with_name(self.conn, self.inner.clone(), name.into())
    }

    /// Get the current setting for what happens to the transaction when it is
    /// dropped.
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
    pub fn commit(mut self) -> Result<()> {
        self.commit_()
    }

    fn commit_(&mut self) -> Result<()> {
        let r = run_managed(&self.inner, || self.conn.execute_batch("COMMIT"));
        match r {
            Ok(()) => {
                self.finished = true;
                if self.conn.is_autocommit() {
                    self.inner.commit_succeeded();
                    self.inner.deliver(self.conn);
                }
                Ok(())
            }
            Err(e) => {
                // Commit failed (e.g. a deferred constraint violation). The
                // transaction is still open; discard the recorded changes so
                // they cannot leak to the callback.
                self.inner.clear();
                Err(e)
            }
        }
    }

    /// A convenience method which consumes and rolls back the transaction.
    pub fn rollback(mut self) -> Result<()> {
        self.rollback_()
    }

    fn rollback_(&mut self) -> Result<()> {
        let r = run_managed(&self.inner, || self.conn.execute_batch("ROLLBACK"));
        self.finished = true;
        if r.is_ok() {
            self.inner.clear();
        }
        r
    }

    /// Consumes the transaction, committing or rolling back according to the
    /// current setting (see `drop_behavior`).
    pub fn finish(mut self) -> Result<()> {
        self.finish_()
    }

    fn finish_(&mut self) -> Result<()> {
        if self.finished || self.conn.is_autocommit() {
            return Ok(());
        }
        match self.drop_behavior {
            DropBehavior::Commit => self.commit_().or_else(|_| self.rollback_()),
            DropBehavior::Rollback => self.rollback_(),
            DropBehavior::Ignore => Ok(()),
            DropBehavior::Panic => panic!("Transaction dropped unexpectedly."),
        }
    }
}

impl Deref for ManagedTransaction<'_> {
    type Target = Connection;

    #[inline]
    fn deref(&self) -> &Connection {
        self.conn
    }
}

#[expect(unused_must_use)]
impl Drop for ManagedTransaction<'_> {
    #[inline]
    fn drop(&mut self) {
        self.finish_();
    }
}

/// A managed savepoint, created by a [`TransactionChangeCollector`].
///
/// Changes made within a managed savepoint are recorded. If the savepoint is
/// rolled back, changes made after it are discarded. If the savepoint is the
/// outermost frame of a transaction started by a savepoint (with no preceding
/// `BEGIN`), releasing it commits the transaction and delivers the recorded
/// changes.
///
/// ## Note
///
/// Savepoints roll back by default. Use [`commit`](ManagedSavepoint::commit) to
/// explicitly release the savepoint, or
/// [`set_drop_behavior`](ManagedSavepoint::set_drop_behavior) to change what
/// happens when the savepoint is dropped.
pub struct ManagedSavepoint<'conn> {
    conn: &'conn Connection,
    inner: Arc<Inner>,
    name: String,
    drop_behavior: DropBehavior,
    finished: bool,
}

impl<'conn> ManagedSavepoint<'conn> {
    fn begin(conn: &'conn Connection, inner: Arc<Inner>) -> Result<Self> {
        let name = inner.state.lock().unwrap().next_savepoint_name();
        let sql = cmd("SAVEPOINT", false, &name)?;
        run_managed(&inner, || conn.execute_batch(sql.as_str()))?;
        {
            let mut s = inner.state.lock().unwrap();
            s.depth += 1;
            s.checkpoint(name.clone());
        }
        Ok(ManagedSavepoint {
            conn,
            inner,
            name,
            drop_behavior: DropBehavior::Rollback,
            finished: false,
        })
    }

    fn with_name(conn: &'conn Connection, inner: Arc<Inner>, name: String) -> Result<Self> {
        let sql = cmd("SAVEPOINT", false, &name)?;
        run_managed(&inner, || conn.execute_batch(sql.as_str()))?;
        {
            let mut s = inner.state.lock().unwrap();
            s.depth += 1;
            s.checkpoint(name.clone());
        }
        Ok(ManagedSavepoint {
            conn,
            inner,
            name,
            drop_behavior: DropBehavior::Rollback,
            finished: false,
        })
    }

    /// Begin a nested managed savepoint.
    pub fn savepoint(&mut self) -> Result<ManagedSavepoint<'_>> {
        ManagedSavepoint::begin(self.conn, self.inner.clone())
    }

    /// Begin a nested managed savepoint with a user-provided name.
    pub fn savepoint_with_name<T: Into<String>>(
        &mut self,
        name: T,
    ) -> Result<ManagedSavepoint<'_>> {
        ManagedSavepoint::with_name(self.conn, self.inner.clone(), name.into())
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

    /// A convenience method which consumes and releases the savepoint.
    pub fn commit(mut self) -> Result<()> {
        self.commit_()
    }

    fn commit_(&mut self) -> Result<()> {
        let sql = cmd("RELEASE", false, &self.name)?;
        let r = run_managed(&self.inner, || self.conn.execute_batch(sql.as_str()));
        match r {
            Ok(()) => {
                self.finished = true;
                {
                    let mut s = self.inner.state.lock().unwrap();
                    s.depth = s.depth.saturating_sub(1);
                    s.release(&self.name);
                }
                if self.conn.is_autocommit() {
                    self.inner.commit_succeeded();
                    self.inner.deliver(self.conn);
                }
                Ok(())
            }
            Err(e) => {
                // RELEASE failed; the savepoint is still active. Discard the
                // recorded changes so they cannot leak.
                self.inner.clear();
                Err(e)
            }
        }
    }

    /// Roll back to the savepoint.
    ///
    /// Changes made after the savepoint was taken are discarded. The savepoint
    /// remains active and can be rolled back to again or released.
    pub fn rollback(&mut self) -> Result<()> {
        let sql = cmd("ROLLBACK", true, &self.name)?;
        let r = run_managed(&self.inner, || self.conn.execute_batch(sql.as_str()));
        if r.is_ok() {
            self.inner.state.lock().unwrap().rollback_to(&self.name);
        }
        r
    }

    /// Consumes the savepoint, committing or rolling back according to the
    /// current setting (see `drop_behavior`).
    pub fn finish(mut self) -> Result<()> {
        self.finish_()
    }

    fn finish_(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        match self.drop_behavior {
            DropBehavior::Commit => self
                .commit_()
                .or_else(|_| self.rollback().and_then(|()| self.commit_())),
            DropBehavior::Rollback => self.rollback().and_then(|()| self.commit_()),
            DropBehavior::Ignore => Ok(()),
            DropBehavior::Panic => panic!("Savepoint dropped unexpectedly."),
        }
    }
}

impl Deref for ManagedSavepoint<'_> {
    type Target = Connection;

    #[inline]
    fn deref(&self) -> &Connection {
        self.conn
    }
}

#[expect(unused_must_use)]
impl Drop for ManagedSavepoint<'_> {
    #[inline]
    fn drop(&mut self) {
        self.finish_();
    }
}
