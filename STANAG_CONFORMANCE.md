# STANAG 4609 / MISB ST 0601 Conformance Checklist

This checklist is for Cairn-Edge evidence export validation on Jetson Orin Nano. The exporter is intentionally off the hot path and should run at no more than 1 Hz.

## Required metadata coverage

- [ ] MISB ST 0601 local set present in MPEG-TS output.
- [ ] Tag 2 `PrecisionTimeStamp` present and UTC-based.
- [ ] Tag 13 frame corner coordinates present for all four corners.
- [ ] Tag 14 camera/viewpoint latitude and longitude present.
- [ ] Tag 15 platform heading angle present.
- [ ] Tag 65 sensor relative azimuth/elevation present where available.
- [ ] Tag 17 slant range present when a calibrated range estimate exists.
- [ ] Warning flag is emitted when GPS/PPS/PTP precision time is unavailable.

## Video/container validation

- [ ] MPEG-TS container opens in ffprobe.
- [ ] H.265 stream is present.
- [ ] KLV/private metadata stream is present in-band, not only as a sidecar.
- [ ] Timestamp cadence is <=1 Hz.
- [ ] Export remains asynchronous and does not block the tracking loop.

## Suggested validation commands

```bash
ffprobe -hide_banner -show_streams evidence/stanag4609/output.ts
ffmpeg -i evidence/stanag4609/output.ts -map 0 -c copy -f null -
exiftool evidence/stanag4609/output.ts
```

## Jetson Orin Nano constraints

- [ ] Export disabled by default in `configs/thermal.yaml`.
- [ ] Software H.265 encode uses `libx265` with `ultrafast` preset.
- [ ] CPU overhead measured at <=5% per exported stream during 1 Hz snapshots.
- [ ] Thermal fusion active power increase measured near +1.5W or documented.
- [ ] Thermal fusion latency measured below 15 ms/frame at configured cadence.

## Degraded-mode validation

- [ ] Thermal camera disconnected: RGB-only mode continues and health is degraded.
- [ ] GPS/PPS/PTP unavailable: system time is used with warning flag.
- [ ] Disk full or permission failure: exporter retries 3 times, disables itself, and reports health error.

## Notes

The Python scaffold builds MISB-style local set bytes and asynchronous video snapshots. True in-band KLV muxing depends on the final GStreamer or FFmpeg build available on the deployment image, so it must be validated on target hardware before claiming STANAG 4609 conformance.
