//! Acceptance tests for the transaction change collector.

use rusqlite::hooks::txn_changes::{TransactionChange, TransactionChangeCollector};
use rusqlite::hooks::Action;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

fn setup() -> (Connection, Arc<Mutex<Vec<TransactionChange>>>) {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch("CREATE TABLE foo (id INTEGER PRIMARY KEY, v TEXT)")
        .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    (db, events)
}

fn make_collector(
    db: &mut Connection,
    events: &Arc<Mutex<Vec<TransactionChange>>>,
) -> TransactionChangeCollector {
    let events = events.clone();
    TransactionChangeCollector::new(db, move |_conn, changes: &[TransactionChange]| {
        events.lock().unwrap().extend_from_slice(changes);
    })
    .unwrap()
}

#[test]
fn insert_savepoint_rollback_commit() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    let mut tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    {
        let mut sp = tx.savepoint().unwrap();
        sp.execute_batch("UPDATE foo SET v = 'b' WHERE id = 1")
            .unwrap();
        sp.rollback().unwrap();
        sp.commit().unwrap();
    }
    tx.commit().unwrap();

    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 1, "only the retained insert should be delivered");
    assert_eq!(ev[0].action, Action::SQLITE_INSERT);
    assert_eq!(ev[0].database, "main");
    assert_eq!(ev[0].table, "foo");
    assert_eq!(ev[0].rowid, 1);
}

#[test]
fn nested_release() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    let mut tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    {
        let mut sp1 = tx.savepoint().unwrap();
        sp1.execute_batch("INSERT INTO foo (v) VALUES ('b')").unwrap();
        {
            let sp2 = sp1.savepoint().unwrap();
            sp2.execute_batch("INSERT INTO foo (v) VALUES ('c')").unwrap();
            sp2.commit().unwrap();
        }
        sp1.commit().unwrap();
    }
    tx.commit().unwrap();

    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 3);
    assert_eq!(ev[0].action, Action::SQLITE_INSERT);
    assert_eq!(ev[1].action, Action::SQLITE_INSERT);
    assert_eq!(ev[2].action, Action::SQLITE_INSERT);
    assert_eq!(ev[0].rowid, 1);
    assert_eq!(ev[1].rowid, 2);
    assert_eq!(ev[2].rowid, 3);
}

#[test]
fn outermost_rollback_discards() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    let tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    tx.rollback().unwrap();

    assert!(events.lock().unwrap().is_empty());
    assert!(db.is_autocommit());
}

#[test]
fn constraint_commit_failure_does_not_leak() {
    let mut db = Connection::open_in_memory().unwrap();
    db.execute_batch(
        "CREATE TABLE parent (id INTEGER PRIMARY KEY);
         CREATE TABLE child (id INTEGER REFERENCES parent(id) DEFERRABLE INITIALLY DEFERRED);",
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let collector = make_collector(&mut db, &events);

    let tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO child VALUES (999)").unwrap();
    let r = tx.commit();
    assert!(r.is_err(), "deferred FK should fail commit");

    // The failed transaction must not have delivered any events.
    assert!(events.lock().unwrap().is_empty());

    // The transaction was rolled back by drop; a subsequent valid transaction
    // works and delivers correctly.
    let tx2 = collector.transaction(&mut db).unwrap();
    tx2.execute_batch("INSERT INTO parent VALUES (1)").unwrap();
    tx2.execute_batch("INSERT INTO child VALUES (1)").unwrap();
    tx2.commit().unwrap();

    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 2);
    assert_eq!(ev[0].table, "parent");
    assert_eq!(ev[1].table, "child");
}

#[test]
fn raw_sql_transaction_control_rejected() {
    let (mut db, _events) = setup();
    let collector = make_collector(&mut db, &_events);

    for sql in [
        "BEGIN",
        "BEGIN DEFERRED",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT x",
        "RELEASE SAVEPOINT x",
        "ROLLBACK TO SAVEPOINT x",
    ] {
        assert!(
            db.execute_batch(sql).is_err(),
            "raw SQL `{sql}` should be rejected"
        );
    }

    // Non-transaction statements still work.
    db.execute_batch("INSERT INTO foo (v) VALUES ('x')").unwrap();

    // Managed transactions still work.
    let tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('y')").unwrap();
    tx.commit().unwrap();
}

#[test]
fn callback_reentrant_query() {
    let (mut db, events) = setup();
    let query_count = Arc::new(Mutex::new(0i64));
    let collector = TransactionChangeCollector::new(&mut db, {
        let events = events.clone();
        let query_count = query_count.clone();
        move |conn, changes: &[TransactionChange]| {
            // Reentrant query: the callback can query the database now that the
            // outermost transaction has committed.
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM foo", [], |row| row.get(0))
                .unwrap();
            *query_count.lock().unwrap() = count;
            events.lock().unwrap().extend_from_slice(changes);
        }
    })
    .unwrap();

    let tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('b')").unwrap();
    tx.commit().unwrap();

    assert_eq!(*query_count.lock().unwrap(), 2);
    assert_eq!(events.lock().unwrap().len(), 2);
}

#[test]
fn savepoint_started_transaction_commits() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    // No BEGIN: the savepoint starts a transaction; RELEASE commits it.
    let sp = collector.savepoint(&mut db).unwrap();
    sp.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    sp.commit().unwrap();

    assert!(db.is_autocommit());
    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].action, Action::SQLITE_INSERT);
}

#[test]
fn update_and_delete_recorded() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    let tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    tx.execute_batch("UPDATE foo SET v = 'b' WHERE id = 1")
        .unwrap();
    tx.execute_batch("DELETE FROM foo WHERE id = 1").unwrap();
    tx.commit().unwrap();

    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 3);
    assert_eq!(ev[0].action, Action::SQLITE_INSERT);
    assert_eq!(ev[1].action, Action::SQLITE_UPDATE);
    assert_eq!(ev[2].action, Action::SQLITE_DELETE);
    // rowids are preserved in order
    assert_eq!(ev[0].rowid, 1);
    assert_eq!(ev[1].rowid, 1);
    assert_eq!(ev[2].rowid, 1);
}

#[test]
fn no_events_outside_transaction() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    // Autocommit statements are not recorded.
    db.execute_batch("INSERT INTO foo (v) VALUES ('a')").unwrap();
    db.execute_batch("UPDATE foo SET v = 'b' WHERE id = 1")
        .unwrap();

    assert!(events.lock().unwrap().is_empty());

    // A managed transaction still delivers.
    let tx = collector.transaction(&mut db).unwrap();
    tx.execute_batch("INSERT INTO foo (v) VALUES ('c')").unwrap();
    tx.commit().unwrap();

    assert_eq!(events.lock().unwrap().len(), 1);
}

#[test]
fn multiple_transactions_sequential() {
    let (mut db, events) = setup();
    let collector = make_collector(&mut db, &events);

    for i in 0..3 {
        let tx = collector.transaction(&mut db).unwrap();
        tx.execute_batch(&format!("INSERT INTO foo (v) VALUES ('{i}')"))
            .unwrap();
        tx.commit().unwrap();
    }

    let ev = events.lock().unwrap();
    assert_eq!(ev.len(), 3);
    for (i, e) in ev.iter().enumerate() {
        assert_eq!(e.action, Action::SQLITE_INSERT);
        assert_eq!(e.rowid, (i + 1) as i64);
    }
}
