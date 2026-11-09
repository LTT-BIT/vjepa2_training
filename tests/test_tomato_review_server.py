import pytest
from tools.serve_tomato_boundary_review import byte_range


@pytest.mark.parametrize('value,expected',[(None,(0,99,False)),('bytes=0-9',(0,9,True)),('bytes=50-',(50,99,True)),('bytes=-10',(90,99,True)),('bytes=50-999',(50,99,True))])
def test_review_video_seek_uses_correct_byte_bounds(value,expected):
    assert byte_range(value,100)==expected


@pytest.mark.parametrize('value',['bytes=100-','bytes=9-1','bytes=-0','bytes=0-1,4-5','items=1-2'])
def test_rejects_invalid_or_unsupported_video_ranges(value):
    with pytest.raises(ValueError):byte_range(value,100)
