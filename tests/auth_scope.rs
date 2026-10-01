//! Acceptance tests for scoped authorizers (`Connection::authorizer_scope`)
//! and the `ReadOnlyTables` whitelist policy. Everything runs against
//! in-memory databases; no external database is needed.
#![cfg(all(feature = "hooks", feature = "cache"))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rusqlite::hooks::{AuthAction, AuthContext, Authorization, ReadOnlyTables, ScopePolicy as _};
use rusqlite::{Connection, Result};

fn test_db() -> Result<Connection> {
    let db = Connection::open_in_memory()?;
    db.execute_batch(
        "CREATE TABLE public(id INTEGER PRIMARY KEY, name TEXT);
         CREATE TABLE private(id INTEGER PRIMARY KEY, secret TEXT);
         INSERT INTO public(name) VALUES ('a');
         INSERT INTO private(secret) VALUES ('s');
         CREATE VIEW v_public AS SELECT id, name FROM public;
         CREATE VIEW v_private AS SELECT id, secret FROM private;",
    )?;
    Ok(db)
}

#[test]
fn scoped_read_only_allows_denies_and_restores() -> Result<()> {
    let db = test_db()?;

    // Outside any scope the connection is fully writable.
    db.execute("INSERT INTO public(name) VALUES ('before')", [])?;
    println!("[允许] 作用域外: 可写 public 表");

    {
        let _scope = db.authorizer_scope(ReadOnlyTables::new().allow_table("public"))?;

        let n: i64 = db.query_row("SELECT count(*) FROM public", [], |r| r.get(0))?;
        println!("[允许] 作用域内: 直接读取 public 表 ({n} 行)");

        let n: i64 = db.query_row("SELECT count(*) FROM v_public", [], |r| r.get(0))?;
        println!("[允许] 作用域内: 通过 view 读取 public 表 ({n} 行)");

        let one: i64 = db.query_row("SELECT abs(-1)", [], |r| r.get(0))?;
        assert_eq!(one, 1);
        println!("[允许] 作用域内: 调用白名单内建函数 abs");

        db.execute_batch("PRAGMA table_info(public)")?;
        println!("[允许] 作用域内: 只读 PRAGMA table_info");

        assert!(db.execute_batch("SELECT * FROM private").is_err());
        println!("[拒绝] 作用域内: 直接读取 private 表");

        assert!(db.execute_batch("SELECT * FROM v_private").is_err());
        println!("[拒绝] 作用域内: 通过 view 读取 private 表");

        // Not detectable by looking at the first word of the SQL string:
        // the read of `private` hides behind a CTE.
        assert!(
            db.execute_batch("WITH x AS (SELECT * FROM private) SELECT * FROM x")
                .is_err()
        );
        println!("[拒绝] 作用域内: 通过 CTE 读取 private 表(不靠 SQL 首词判断)");

        assert!(
            db.execute_batch("INSERT INTO public(name) VALUES ('x')")
                .is_err()
        );
        assert!(db.execute_batch("UPDATE public SET name = 'y'").is_err());
        assert!(db.execute_batch("DELETE FROM public").is_err());
        println!("[拒绝] 作用域内: 写入/更新/删除 public 表");

        assert!(db.execute_batch("CREATE TABLE evil(x)").is_err());
        assert!(db.execute_batch("DROP TABLE public").is_err());
        println!("[拒绝] 作用域内: DDL (CREATE/DROP)");

        assert!(db.execute_batch("ATTACH ':memory:' AS other").is_err());
        println!("[拒绝] 作用域内: ATTACH");

        assert!(db.execute_batch("PRAGMA writable_schema = 1").is_err());
        assert!(db.execute_batch("PRAGMA user_version = 1").is_err());
        assert!(db.execute_batch("PRAGMA foreign_keys = ON").is_err());
        println!("[拒绝] 作用域内: 危险 PRAGMA (writable_schema/user_version/foreign_keys)");

        assert!(db.execute_batch("SELECT load_extension('x')").is_err());
        assert!(db.execute_batch("SELECT changes()").is_err());
        println!("[拒绝] 作用域内: 未授权函数 (load_extension/changes)");
    }

    // The previous authorizer (none) is restored on scope exit.
    db.execute("INSERT INTO public(name) VALUES ('after')", [])?;
    db.execute_batch("SELECT * FROM private")?;
    println!("[恢复] 退出作用域后: 恢复可写、可读 private");
    Ok(())
}

#[test]
fn nested_scopes_intersect_whitelists() -> Result<()> {
    let db = test_db()?;
    db.execute_batch(
        "CREATE TABLE a(x); CREATE TABLE b(x); CREATE TABLE c(x);
         INSERT INTO a VALUES (1); INSERT INTO b VALUES (1); INSERT INTO c VALUES (1);",
    )?;

    let outer = db.authorizer_scope(ReadOnlyTables::new().allow_tables(["a", "b"]))?;
    assert!(db.execute_batch("SELECT * FROM a").is_ok());
    assert!(db.execute_batch("SELECT * FROM b").is_ok());
    assert!(db.execute_batch("SELECT * FROM c").is_err());
    println!("[允许/拒绝] 外层作用域 a,b: a、b 可读, c 拒绝");

    {
        let _inner = db.authorizer_scope(ReadOnlyTables::new().allow_tables(["b", "c"]))?;
        // Intersection of {a, b} and {b, c} is {b}.
        assert!(db.execute_batch("SELECT * FROM a").is_err());
        assert!(db.execute_batch("SELECT * FROM b").is_ok());
        assert!(db.execute_batch("SELECT * FROM c").is_err());
        println!("[交集] 嵌套内层 b,c: 仅交集 b 可读, a、c 均拒绝");
    }

    // Leaving the inner scope restores the outer policy.
    assert!(db.execute_batch("SELECT * FROM a").is_ok());
    assert!(db.execute_batch("SELECT * FROM c").is_err());
    println!("[恢复] 内层退出后: 恢复外层策略 (a、b 可读)");

    drop(outer);
    db.execute_batch("INSERT INTO c VALUES (2)")?;
    println!("[恢复] 外层退出后: 恢复原始授权器 (可写)");
    Ok(())
}

#[test]
fn authorizer_restored_after_panic() -> Result<()> {
    let db = test_db()?;

    // Keep the test output clean; the panic is expected.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _scope = db
            .authorizer_scope(ReadOnlyTables::new().allow_table("public"))
            .unwrap();
        assert!(
            db.execute_batch("INSERT INTO public(name) VALUES ('no')")
                .is_err()
        );
        println!("[拒绝] panic 前: 作用域内写入被拒绝");
        panic!("boom");
    }));
    let _ = std::panic::take_hook(); // discard the silent hook
    std::panic::set_hook(prev_hook);
    assert!(result.is_err());

    // Unwinding dropped the guard: the previous authorizer is restored.
    db.execute("INSERT INTO public(name) VALUES ('yes')", [])?;
    println!("[恢复] panic 展开后: 授权器已恢复 (可写)");

    // And restore keeps working for subsequent scopes (double-check).
    {
        let _scope = db.authorizer_scope(ReadOnlyTables::new().allow_table("public"))?;
        assert!(db.execute_batch("DELETE FROM public").is_err());
        println!("[拒绝] 再次进入作用域: 删除仍被拒绝");
    }
    db.execute_batch("DELETE FROM public")?;
    println!("[恢复] 再次退出作用域后: 恢复仍然正确 (可删除)");
    Ok(())
}

#[test]
fn cached_statements_cannot_bypass_scope() -> Result<()> {
    let db = test_db()?;

    // Prepare and cache a statement while no authorizer is active.
    {
        let mut stmt = db.prepare_cached("SELECT count(*) FROM private")?;
        let n: i64 = stmt.query_row([], |r| r.get(0))?;
        assert_eq!(n, 1);
        println!("[允许] 作用域外: 预缓存 SELECT private 语句");
    } // idle in the cache now

    let scope = db.authorizer_scope(ReadOnlyTables::new().allow_table("public"))?;

    // The cache was flushed on scope entry: the statement is re-prepared
    // and the new authorizer denies it.
    assert!(db.prepare_cached("SELECT count(*) FROM private").is_err());
    println!("[拒绝] 作用域内: 空闲缓存语句重新准备被拒绝");
    drop(scope);

    // A statement that is still alive while the scope is entered must not
    // bypass the policy either: it is tagged with the old generation and
    // never served from the cache again.
    let stmt = db.prepare_cached("SELECT count(*) FROM private")?;
    let scope = db.authorizer_scope(ReadOnlyTables::new().allow_table("public"))?;
    drop(stmt); // returns to the cache tagged with the old generation
    assert!(db.prepare_cached("SELECT count(*) FROM private").is_err());
    println!("[拒绝] 作用域内: 切换时仍存活的语句回流缓存后不能绕过新策略");
    drop(scope);

    // Restored after the scope ends.
    let mut stmt = db.prepare_cached("SELECT count(*) FROM private")?;
    let n: i64 = stmt.query_row([], |r| r.get(0))?;
    assert_eq!(n, 1);
    println!("[恢复] 退出作用域后: 缓存语句可再次访问 private ({n} 行)");
    Ok(())
}

#[test]
fn previous_authorizer_restored_and_callbacks_counted() -> Result<()> {
    let mut db = test_db()?;

    let base_count = Arc::new(AtomicUsize::new(0));
    let base = Arc::clone(&base_count);
    // Base authorizer: counts every callback and denies DELETE.
    db.authorizer(Some(move |ctx: AuthContext<'_>| {
        base.fetch_add(1, Ordering::SeqCst);
        match ctx.action {
            AuthAction::Delete { .. } => Authorization::Deny,
            _ => Authorization::Allow,
        }
    }))?;

    let scope_count = Arc::new(AtomicUsize::new(0));
    {
        let counter = Arc::clone(&scope_count);
        let policy = ReadOnlyTables::new().allow_table("public");
        let _scope = db.authorizer_scope(move |ctx: AuthContext<'_>| {
            counter.fetch_add(1, Ordering::SeqCst);
            policy.authorize(ctx)
        })?;

        assert!(db.execute_batch("SELECT * FROM public").is_ok());
        assert!(db.execute_batch("SELECT * FROM private").is_err()); // scope denies
        assert!(db.execute_batch("DELETE FROM public").is_err()); // both deny
    }

    let base_in_scope = base_count.load(Ordering::SeqCst);
    let scoped = scope_count.load(Ordering::SeqCst);
    assert!(
        base_in_scope > 0,
        "base hook must keep firing inside the scope"
    );
    assert!(scoped > 0, "scope hook must fire inside the scope");
    println!("[计数] 作用域内: 基础授权器回调 {base_in_scope} 次, 作用域策略回调 {scoped} 次");

    // After the scope, the base authorizer is still in effect...
    assert!(db.execute_batch("DELETE FROM public").is_err());
    println!("[恢复] 退出作用域后: 原授权器仍然生效 (DELETE 仍被拒绝)");

    // ...while the scoped restriction is gone.
    db.execute_batch("SELECT * FROM private")?;
    println!("[恢复] 退出作用域后: 作用域限制已移除 (可读 private)");

    let before = base_count.load(Ordering::SeqCst);
    db.execute_batch("SELECT * FROM public")?;
    assert!(base_count.load(Ordering::SeqCst) > before);
    assert_eq!(
        scope_count.load(Ordering::SeqCst),
        scoped,
        "scoped hook must not fire after its scope ended"
    );
    println!("[计数] 作用域外: 仅基础授权器继续回调");
    Ok(())
}

#[test]
fn view_and_trigger_indirect_access_denied() -> Result<()> {
    let db = test_db()?;
    // A trigger whose body writes `private` when the view is inserted into.
    db.execute_batch(
        "CREATE VIEW v_insert AS SELECT id, name FROM public;
         CREATE TRIGGER trg INSTEAD OF INSERT ON v_insert
         BEGIN
             INSERT INTO private(secret) VALUES (NEW.name);
         END;",
    )?;

    let _scope = db.authorizer_scope(ReadOnlyTables::new().allow_table("public"))?;

    // The actual access performed by the trigger body (insert into
    // `private`) is checked, not just the top-level statement.
    assert!(
        db.execute_batch("INSERT INTO v_insert(name) VALUES ('x')")
            .is_err()
    );
    println!("[拒绝] 作用域内: 通过 view/trigger 间接写入 private");

    // Reading `private` through a view is equally denied (see also the
    // main acceptance test); the authorizer reports the underlying table.
    assert!(db.execute_batch("SELECT * FROM v_private").is_err());
    println!("[拒绝] 作用域内: 通过 view 间接读取 private");
    Ok(())
}
