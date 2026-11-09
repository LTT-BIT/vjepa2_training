from tools.evaluate_tomato_independent_recordings import summarize


def test_event_precision_recall_and_false_per_minute_use_same_duration():
    row=dict(boundaries=[dict(detected=True,previous=3,label=0,offset_s=-.5),dict(detected=False,previous=0,label=1,offset_s=None)],false_events=[dict(time=10.,previous=1,label=2)],duration_s=60.)
    x=summarize({'recording':{'future':row}},'future')
    assert x['precision']==.5 and x['recall']==.5 and x['f1']==.5
    assert x['false_per_minute']==1. and x['offset_p50_s']==-.5
    assert x['by_transition']['adjust->pickup']==dict(total=1,detected=1)
