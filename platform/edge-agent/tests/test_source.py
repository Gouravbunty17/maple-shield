from edge_agent.source import MockSource


def test_mock_source_yields_n_frames():
    s = MockSource(n_frames=5, w=64, h=48)
    frames = list(s.frames())
    assert len(frames) == 5
    i, frame = frames[0]
    assert frame.shape == (48, 64, 3)
    assert i == 0
