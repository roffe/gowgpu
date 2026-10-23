//go:build !(js && wasm)

package vulkan

import (
	"fmt"
	"sync/atomic"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// QuerySet implements hal.QuerySet for Vulkan.
type QuerySet struct {
	pool      vk.QueryPool
	device    *Device
	queryType hal.QueryType
	count     uint32
	// written are the queries a command buffer has written, which a
	// resolve copies; the rest resolve to zero, as Dawn does. A query is
	// reset ahead of each write, so the others are never initialised.
	written []atomic.Bool
}

// write records that query index is written.
func (q *QuerySet) write(index uint32) {
	if index < uint32(len(q.written)) {
		q.written[index].Store(true)
	}
}

// Destroy releases the Vulkan query pool.
func (q *QuerySet) Destroy() {
	if q.pool != 0 && q.device != nil {
		q.device.live.queryPools.remove(q.pool)
		q.device.cmds.DestroyQueryPool(q.device.handle, q.pool, nil)
		q.pool = 0
	}
}

// CreateQuerySet creates a Vulkan query pool.
func (d *Device) CreateQuerySet(desc *hal.QuerySetDescriptor) (hal.QuerySet, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: query set descriptor is nil in Vulkan.CreateQuerySet — core validation gap")
	}

	if desc.Count == 0 {
		return nil, fmt.Errorf("BUG: query set count is 0 in Vulkan.CreateQuerySet — core validation gap")
	}

	var vkQueryType vk.QueryType
	switch desc.Type {
	case hal.QueryTypeTimestamp:
		vkQueryType = vk.QueryTypeTimestamp
	case hal.QueryTypeOcclusion:
		vkQueryType = vk.QueryTypeOcclusion
	default:
		return nil, fmt.Errorf("vulkan: unsupported query type: %d", desc.Type)
	}

	createInfo := vk.QueryPoolCreateInfo{
		SType:      vk.StructureTypeQueryPoolCreateInfo,
		QueryType:  vkQueryType,
		QueryCount: desc.Count,
	}

	var pool vk.QueryPool
	result := d.cmds.CreateQueryPool(d.handle, &createInfo, nil, &pool)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateQueryPool failed: %d", result)
	}

	qs := &QuerySet{
		pool:      pool,
		device:    d,
		queryType: desc.Type,
		count:     desc.Count,
		written:   make([]atomic.Bool, desc.Count),
	}
	d.live.queryPools.add(pool)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeQueryPool, uint64(pool), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeQueryPool, uint64(pool), "QueryPool")
	}
	return qs, nil
}

// DestroyQuerySet destroys a Vulkan query set.
func (d *Device) DestroyQuerySet(querySet hal.QuerySet) {
	if qs, ok := querySet.(*QuerySet); ok {
		qs.Destroy()
	}
}
