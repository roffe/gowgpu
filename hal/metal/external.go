//go:build darwin && !(js && wasm)

package metal

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// MetalDevice returns the id<MTLDevice>, for code sharing the device with
// gowgpu: a video decoder, or another API filling resources it reads.
func (d *Device) MetalDevice() uintptr {
	return uintptr(d.raw)
}

// externalSync is a point on an MTLSharedEvent of code sharing the device.
type externalSync struct {
	event        ID
	wait, signal uint64
}

// externalCopy is a copy out of a resource made outside gowgpu: the planes
// of an IOSurface, or a picture in an MTLBuffer whose rows are bytesPerRow
// apart. Metal orders the passes of a command buffer by the events it waits
// for and signals, not those of other command buffers, so Submit encodes the
// copies in one of their own, between the points AddExternalSync was given.
type externalCopy struct {
	src         ID
	bytesPerRow uint32 // 0 for an IOSurface
	dsts        [3]*Texture
}

// CopyExternalPlanes copies the planes of src, an IOSurfaceRef such as the
// one behind a VideoToolbox CVPixelBuffer, into dsts, plane i into dsts[i], as
// much of it as they hold. Each plane is read as its dst's format, which has
// the plane's bytes per element. An IOSurface has one layer and no layout, so
// slice is ignored and layout returned as it is. The copy runs in the next
// Submit, after the events AddExternalSync was given reach their values.
func (e *CommandEncoder) CopyExternalPlanes(src uintptr, layout, _ uint32, dsts []hal.Texture) uint32 {
	if e.cmdBuffer == 0 || src == 0 || len(dsts) > 3 {
		return layout
	}
	c := externalCopy{src: ID(src)}
	for i, dst := range dsts {
		c.dsts[i], _ = dst.(*Texture)
	}
	e.external = append(e.external, c)
	return layout
}

// CopyExternalBuffer copies src, an id<MTLBuffer> made outside gowgpu that
// another API fills, into dst whole, its rows bytesPerRow apart. Like
// CopyExternalPlanes it runs in the next Submit, after the events.
func (e *CommandEncoder) CopyExternalBuffer(src uintptr, bytesPerRow uint32, dst hal.Texture) {
	t, ok := dst.(*Texture)
	if e.cmdBuffer == 0 || src == 0 || bytesPerRow == 0 || !ok || t == nil {
		return
	}
	e.external = append(e.external, externalCopy{src: ID(src), bytesPerRow: bytesPerRow, dsts: [3]*Texture{t}})
}

// ExternalSyncFull reports false: a submit takes any number of points of
// AddExternalSync.
func (q *Queue) ExternalSyncFull() bool {
	return false
}

// AddExternalSync has the external copies of the next Submit wait for event,
// an id<MTLSharedEvent> of code sharing the device, to reach wait, and signal
// signal once they are done. The command buffers reading what they copied
// run after them, as Metal tracks the textures they write.
func (q *Queue) AddExternalSync(event uintptr, wait, signal uint64) error {
	if event == 0 {
		return fmt.Errorf("metal: AddExternalSync: no event")
	}
	q.external = append(q.external, externalSync{ID(event), wait, signal})
	return nil
}

// commitExternal commits a command buffer with the external copies of
// commandBuffers between the waits and signals AddExternalSync was given,
// ahead of commandBuffers. It is called inside Submit's autorelease pool.
func (q *Queue) commitExternal(commandBuffers []hal.CommandBuffer) error {
	copies := false
	for _, buf := range commandBuffers {
		if cb, ok := buf.(*CommandBuffer); ok && cb != nil && len(cb.external) > 0 {
			copies = true
		}
	}
	if !copies && len(q.external) == 0 {
		return nil
	}
	defer func() { q.external = q.external[:0] }()

	raw := MsgSend(q.commandQueue, Sel("commandBuffer"))
	if raw == 0 {
		return fmt.Errorf("metal: failed to create the command buffer of external copies")
	}
	for _, p := range q.external {
		_ = MsgSend(raw, Sel("encodeWaitForEvent:value:"), uintptr(p.event), uintptr(p.wait))
	}
	if copies {
		blit := MsgSend(raw, Sel("blitCommandEncoder"))
		for _, buf := range commandBuffers {
			if cb, ok := buf.(*CommandBuffer); ok && cb != nil {
				for _, c := range cb.external {
					q.device.encodeExternalCopy(blit, c)
				}
				cb.external = nil
			}
		}
		_ = MsgSend(blit, Sel("endEncoding"))
	}
	for _, p := range q.external {
		_ = MsgSend(raw, Sel("encodeSignalEvent:value:"), uintptr(p.event), uintptr(p.signal))
	}
	_ = MsgSend(raw, Sel("commit"))
	return nil
}

func (d *Device) encodeExternalCopy(blit ID, c externalCopy) {
	if c.bytesPerRow != 0 {
		t := c.dsts[0]
		size := MTLSize{Width: NSUInteger(t.width), Height: NSUInteger(t.height), Depth: 1}
		msgSendVoid(blit, Sel("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
			argPointer(uintptr(c.src)), argUint64(0), argUint64(uint64(c.bytesPerRow)), argUint64(uint64(c.bytesPerRow)*uint64(t.height)),
			argStruct(size, mtlSizeType), argPointer(uintptr(t.raw)), argUint64(0), argUint64(0), argStruct(MTLOrigin{}, mtlOriginType))
		return
	}
	planar := MsgSendUint(c.src, Sel("planeCount")) > 0
	for i, t := range c.dsts {
		if t == nil || t.raw == 0 {
			continue
		}
		w, h := MsgSendUint(c.src, Sel("width")), MsgSendUint(c.src, Sel("height"))
		if planar {
			w = MsgSendUint(c.src, Sel("widthOfPlaneAtIndex:"), uintptr(i))
			h = MsgSendUint(c.src, Sel("heightOfPlaneAtIndex:"), uintptr(i))
		}
		plane := d.planeTexture(c.src, i, t.format, w, h)
		if plane == 0 {
			hal.Logger().Warn("metal: external copy: no texture of the plane", "plane", i, "format", t.format)
			continue
		}
		size := MTLSize{Width: NSUInteger(min(w, uint(t.width))), Height: NSUInteger(min(h, uint(t.height))), Depth: 1}
		msgSendVoid(blit, Sel("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
			argPointer(uintptr(plane)), argUint64(0), argUint64(0), argStruct(MTLOrigin{}, mtlOriginType), argStruct(size, mtlSizeType),
			argPointer(uintptr(t.raw)), argUint64(0), argUint64(0), argStruct(MTLOrigin{}, mtlOriginType))
		Release(plane) // the command buffer holds it until it is done
	}
}

// planeTexture returns plane i of surface, w by h texels of format, or 0.
func (d *Device) planeTexture(surface ID, i int, format gputypes.TextureFormat, w, h uint) ID {
	desc := MsgSend(ID(GetClass("MTLTextureDescriptor")), Sel("new"))
	if desc == 0 {
		return 0
	}
	defer Release(desc)
	_ = MsgSend(desc, Sel("setPixelFormat:"), uintptr(d.adapter.mapTextureFormat(format)))
	_ = MsgSend(desc, Sel("setWidth:"), uintptr(w))
	_ = MsgSend(desc, Sel("setHeight:"), uintptr(h))
	return MsgSend(d.raw, Sel("newTextureWithDescriptor:iosurface:plane:"), uintptr(desc), uintptr(surface), uintptr(i))
}
