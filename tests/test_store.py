from spl_ft.store import Store


def test_queue_priority_dependency_cancel_and_recovery(tmp_path):
    s=Store(tmp_path)
    a=s.enqueue('train',{'name':'first'})
    b=s.enqueue('benchmark',{'name':'depends'},dependency=a['id'],priority=100)
    c=s.enqueue('doctor',{'name':'priority'},priority=20)
    assert s.claim()['id']==c['id']
    s.update(c['id'],state='succeeded')
    assert s.claim()['id']==a['id']
    s.recover()
    assert s.get(a['id'])['state']=='interrupted'
    assert s.claim() is None
    assert s.get(b['id'])['state']=='blocked'


def test_pause_and_fifo(tmp_path):
    s=Store(tmp_path)
    a=s.enqueue('doctor',{'name':'a'})
    s.set_setting('paused',True)
    assert s.claim() is None
    s.set_setting('paused',False)
    assert s.claim()['id']==a['id']
    assert s.claim() is None
