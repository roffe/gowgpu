# Security

gowgpu has no releases yet, so fixes land on `main`.

## Reporting a problem

Don't open a public issue for it. Report it privately through [GitHub's security advisories](https://github.com/roffe/gowgpu/security/advisories/new) and i'll get back to you as soon as i can.

## Worth knowing

gowgpu runs shaders and buffer contents you give it on the GPU driver, and it calls that driver through cgo and `syscall`. A bug in validation can hand a driver something it doesn't expect, so treat untrusted WGSL and untrusted sizes and offsets like any other untrusted input: check them before they reach the device.
