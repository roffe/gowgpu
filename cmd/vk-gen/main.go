// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Command vk-gen generates Pure Go Vulkan bindings from vk.xml specification.
//
// Usage:
//
//	vk-gen -spec vk.xml -out ../hal/vulkan/vk/
//
//nolint:errcheck,gosec,gocritic,goconst,maintidx,funlen,gocyclo,cyclop,gocognit,nestif // code generator
package main

import (
	"bytes"
	"encoding/xml"
	"flag"
	"fmt"
	"go/format"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

var (
	specFile  = flag.String("spec", "vk.xml", "Path to vk.xml specification")
	outputDir = flag.String("out", "../../hal/vulkan/vk/", "Output directory")
	// apiVersion reserved for future multi-API support (e.g., vulkansc)
)

func main() {
	flag.Parse()

	fmt.Printf("vk-gen: Generating Pure Go Vulkan bindings\n")
	fmt.Printf("  Spec: %s\n", *specFile)
	fmt.Printf("  Output: %s\n", *outputDir)

	// Parse vk.xml
	registry, err := parseSpec(*specFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing spec: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("  Types: %d\n", len(registry.Types.Types))
	fmt.Printf("  Enums: %d\n", len(registry.Enums))
	fmt.Printf("  Commands: %d\n", len(registry.Commands.Commands))

	// Create output directory
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output dir: %v\n", err)
		os.Exit(1)
	}

	// Generate files
	if err := generateConstants(registry, *outputDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating constants: %v\n", err)
		os.Exit(1)
	}

	if err := generateTypes(registry, *outputDir, nil); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating types: %v\n", err)
		os.Exit(1)
	}
	pads, err := align8Pads(*outputDir, bitfieldStructs(registry))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error laying out structs for 32-bit targets: %v\n", err)
		os.Exit(1)
	}
	if err := generateTypes(registry, *outputDir, pads); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating types: %v\n", err)
		os.Exit(1)
	}

	if err := generateCommands(registry, *outputDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating commands: %v\n", err)
		os.Exit(1)
	}

	if err := generateLoader(registry, *outputDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating loader: %v\n", err)
		os.Exit(1)
	}

	// Format generated files with gofmt.
	for _, name := range []string{"const_gen.go", "types_gen.go", "types_align4_gen.go", "types_align8_gen.go", "commands_gen.go"} {
		path := filepath.Join(*outputDir, name)
		src, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %s for formatting: %v\n", name, err)
			os.Exit(1)
		}
		formatted, err := format.Source(src)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error formatting %s: %v\n", name, err)
			os.Exit(1)
		}
		if err := os.WriteFile(path, formatted, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing formatted %s: %v\n", name, err)
			os.Exit(1)
		}
	}

	fmt.Println("Generation complete!")
}

// Registry represents the root of vk.xml
type Registry struct {
	XMLName    xml.Name      `xml:"registry"`
	Types      TypesSection  `xml:"types"`
	Enums      []EnumsGroup  `xml:"enums"`
	Commands   CommandsGroup `xml:"commands"`
	Features   []Feature     `xml:"feature"`
	Extensions Extensions    `xml:"extensions"`
}

// Extensions contains all Vulkan extensions
type Extensions struct {
	Extensions []Extension `xml:"extension"`
}

// Extension represents a Vulkan extension
type Extension struct {
	Name      string             `xml:"name,attr"`
	Number    int                `xml:"number,attr"`
	Type      string             `xml:"type,attr"`
	Platform  string             `xml:"platform,attr"`
	Supported string             `xml:"supported,attr"`
	Requires  []ExtensionRequire `xml:"require"`
}

// ExtensionRequire contains required types and enums for an extension
type ExtensionRequire struct {
	Enums []ExtensionEnum `xml:"enum"`
	Types []ExtensionType `xml:"type"`
}

// ExtensionEnum is an enum value added by an extension
type ExtensionEnum struct {
	Name    string `xml:"name,attr"`
	Value   string `xml:"value,attr"`
	Offset  string `xml:"offset,attr"`
	Extends string `xml:"extends,attr"`
	Bitpos  string `xml:"bitpos,attr"`
	Dir     string `xml:"dir,attr"` // "-" for negative values
}

// ExtensionType is a type reference in an extension
type ExtensionType struct {
	Name string `xml:"name,attr"`
}

type TypesSection struct {
	Types []Type `xml:"type"`
}

type Type struct {
	Name      string   `xml:"name,attr"`
	Category  string   `xml:"category,attr"`
	Alias     string   `xml:"alias,attr"`
	Parent    string   `xml:"parent,attr"`
	Members   []Member `xml:"member"`
	InnerName string   `xml:"name"` // For types where name is element content
	InnerType string   `xml:"type"` // The typedef'd type, VkFlags or VkFlags64 for a bitmask
	Requires  string   `xml:"requires,attr"`
}

type Member struct {
	Name     string `xml:"name"`
	Type     string `xml:"type"`
	Enum     string `xml:"enum"` // Array size constant
	Values   string `xml:"values,attr"`
	Len      string `xml:"len,attr"`
	Optional string `xml:"optional,attr"`
	RawXML   string `xml:",innerxml"`
}

type EnumsGroup struct {
	Name     string `xml:"name,attr"`
	Type     string `xml:"type,attr"`
	BitWidth string `xml:"bitwidth,attr"` // "64" for the FlagBits of a VkFlags64 bitmask
	Comment  string `xml:"comment,attr"`
	Enums    []Enum `xml:"enum"`
}

type Enum struct {
	Name    string `xml:"name,attr"`
	Value   string `xml:"value,attr"`
	Bitpos  string `xml:"bitpos,attr"`
	Alias   string `xml:"alias,attr"`
	Comment string `xml:"comment,attr"`
}

type CommandsGroup struct {
	Commands []Command `xml:"command"`
}

type Command struct {
	Alias        string       `xml:"alias,attr"`
	Name         string       `xml:"name,attr"`
	Proto        CommandProto `xml:"proto"`
	Params       []Param      `xml:"param"`
	SuccessCodes string       `xml:"successcodes,attr"`
	ErrorCodes   string       `xml:"errorcodes,attr"`
}

type CommandProto struct {
	Type string `xml:"type"`
	Name string `xml:"name"`
}

type Param struct {
	Name     string `xml:"name"`
	Type     string `xml:"type"`
	Len      string `xml:"len,attr"`
	Optional string `xml:"optional,attr"`
	RawXML   string `xml:",innerxml"`
	API      string `xml:"api,attr"` // vulkan, vulkansc, vulkanbase - filter out vulkansc
}

type Feature struct {
	API     string    `xml:"api,attr"`
	Name    string    `xml:"name,attr"`
	Number  string    `xml:"number,attr"`
	Require []Require `xml:"require"`
}

type Require struct {
	Types    []RequireType    `xml:"type"`
	Enums    []RequireEnum    `xml:"enum"`
	Commands []RequireCommand `xml:"command"`
}

type RequireType struct {
	Name string `xml:"name,attr"`
}

type RequireEnum struct {
	Name string `xml:"name,attr"`
}

type RequireCommand struct {
	Name string `xml:"name,attr"`
}

func parseSpec(path string) (*Registry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var registry Registry
	if err := xml.Unmarshal(data, &registry); err != nil {
		return nil, err
	}

	return &registry, nil
}

func generateConstants(registry *Registry, outDir string) error {
	f, err := os.Create(filepath.Join(outDir, "const_gen.go"))
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintln(f, "// Code generated by vk-gen. DO NOT EDIT.")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "package vk")
	fmt.Fprintln(f, "")

	// Generate VkResult and other enums
	for _, group := range registry.Enums {
		// Handle API Constants separately
		if group.Name == "API Constants" {
			if len(group.Enums) == 0 {
				continue
			}
			fmt.Fprintln(f, "// API Constants")
			fmt.Fprintln(f, "const (")
			for _, e := range group.Enums {
				if e.Alias != "" || e.Value == "" {
					continue
				}
				goName := vkToGoConst(e.Name)
				goValue := convertCValue(e.Value)
				fmt.Fprintf(f, "\t%s = %s\n", goName, goValue)
			}
			fmt.Fprintln(f, ")")
			fmt.Fprintln(f, "")
			continue
		}

		// Generate enum/bitmask types
		if group.Type == "enum" || group.Type == "bitmask" {
			typeName := vkToGoType(group.Name)
			// int64 for the FlagBits vk.xml marks 64 bits wide, as the name
			// does not tell (VkTensorUsageFlagBitsARM is, the H264 ones are not).
			baseType := "int32"
			if group.BitWidth == "64" {
				baseType = "int64"
			}
			fmt.Fprintf(f, "// %s\n", group.Name)
			fmt.Fprintf(f, "type %s %s\n", typeName, baseType)
			fmt.Fprintln(f, "")

			// Only generate const block if there are values
			// (extension values will be generated separately)
			if len(group.Enums) > 0 {
				fmt.Fprintln(f, "const (")
				for _, e := range group.Enums {
					if e.Alias != "" {
						continue
					}
					goName := vkToGoConst(e.Name)
					if e.Bitpos != "" {
						fmt.Fprintf(f, "\t%s %s = 1 << %s\n", goName, typeName, e.Bitpos)
					} else if e.Value != "" {
						fmt.Fprintf(f, "\t%s %s = %s\n", goName, typeName, e.Value)
					}
				}
				fmt.Fprintln(f, ")")
				fmt.Fprintln(f, "")
			}
		}
	}

	// Collect all already-defined enum names to avoid duplicates
	definedEnums := make(map[string]bool)
	for _, group := range registry.Enums {
		for _, e := range group.Enums {
			if e.Alias == "" {
				definedEnums[vkToGoConst(e.Name)] = true
			}
		}
	}

	// Generate extension enum values grouped by the type they extend.
	// Sort keys for deterministic output (Go map iteration is random).
	extensionEnums := collectExtensionEnums(registry)
	extendedTypes := make([]string, 0, len(extensionEnums))
	for k := range extensionEnums {
		extendedTypes = append(extendedTypes, k)
	}
	sort.Strings(extendedTypes)
	for _, extendsType := range extendedTypes {
		enums := extensionEnums[extendsType]
		goTypeName := vkToGoType(extendsType)

		// Filter out duplicates
		var uniqueEnums []ExtensionEnumValue
		seenNames := make(map[string]bool)
		for _, e := range enums {
			goName := vkToGoConst(e.Name)
			if !definedEnums[goName] && !seenNames[goName] {
				uniqueEnums = append(uniqueEnums, e)
				seenNames[goName] = true
			}
		}

		if len(uniqueEnums) == 0 {
			continue
		}

		fmt.Fprintf(f, "// %s extension values\n", extendsType)
		fmt.Fprintln(f, "const (")
		for _, e := range uniqueEnums {
			goName := vkToGoConst(e.Name)
			fmt.Fprintf(f, "\t%s %s = %d\n", goName, goTypeName, e.Value)
		}
		fmt.Fprintln(f, ")")
		fmt.Fprintln(f, "")
	}

	return nil
}

// ExtensionEnumValue holds a calculated extension enum value
type ExtensionEnumValue struct {
	Name  string
	Value int64
}

// collectExtensionEnums collects all extension enums and calculates their values
func collectExtensionEnums(registry *Registry) map[string][]ExtensionEnumValue {
	result := make(map[string][]ExtensionEnumValue)

	for _, ext := range registry.Extensions.Extensions {
		// Skip unsupported/disabled extensions
		if ext.Supported == "disabled" {
			continue
		}

		for _, req := range ext.Requires {
			for _, e := range req.Enums {
				if e.Extends == "" {
					continue // Not extending an existing enum
				}

				var value int64
				if e.Value != "" {
					// Direct value
					v, err := strconv.ParseInt(e.Value, 0, 64)
					if err != nil {
						continue
					}
					value = v
				} else if e.Offset != "" {
					// Calculate from extension number and offset
					// Formula: base + (ext_number - 1) * 1000 + offset
					// where base = 1000000000 for positive values
					offset, err := strconv.Atoi(e.Offset)
					if err != nil {
						continue
					}
					value = 1000000000 + int64(ext.Number-1)*1000 + int64(offset)
					if e.Dir == "-" {
						value = -value
					}
				} else if e.Bitpos != "" {
					// Bit position
					bitpos, err := strconv.Atoi(e.Bitpos)
					if err != nil {
						continue
					}
					value = 1 << bitpos
				} else {
					continue // No value, skip
				}

				result[e.Extends] = append(result[e.Extends], ExtensionEnumValue{
					Name:  e.Name,
					Value: value,
				})
			}
		}
	}

	return result
}

func generateTypes(registry *Registry, outDir string, pads map[string][]int64) error {
	f, err := os.Create(filepath.Join(outDir, "types_gen.go"))
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintln(f, "// Code generated by vk-gen. DO NOT EDIT.")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "package vk")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "import \"unsafe\"")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "// Prevent unused import error")
	fmt.Fprintln(f, "var _ = unsafe.Sizeof(0)")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "// Base types")
	fmt.Fprintln(f, "type (")
	fmt.Fprintln(f, "\tBool32        uint32")
	fmt.Fprintln(f, "\tDeviceSize    uint64")
	fmt.Fprintln(f, "\tDeviceAddress uint64")
	fmt.Fprintln(f, "\tFlags         uint32")
	fmt.Fprintln(f, "\tFlags64       uint64")
	fmt.Fprintln(f, "\tSampleMask    uint32")
	fmt.Fprintln(f, ")")
	fmt.Fprintln(f, "")

	// Bitmask types (Flags aliases)
	seenBitmasks := make(map[string]bool)
	bitmasks := []struct {
		name     string
		baseType string
	}{}
	for _, t := range registry.Types.Types {
		if t.Category == "bitmask" {
			name := t.Name
			if name == "" {
				name = t.InnerName
			}
			if name == "" || !strings.HasPrefix(name, "Vk") {
				continue
			}
			// Check if it's an alias
			if t.Alias != "" {
				continue
			}
			goName := vkToGoType(name)
			if seenBitmasks[goName] {
				continue
			}
			seenBitmasks[goName] = true
			// Base type as vk.xml typedefs it: VkFlags or VkFlags64. The
			// name says nothing (VkPipelineStageFlags2 is 64-bit,
			// VkVideoEncodeH264StdFlagsKHR is not).
			baseType := "Flags"
			if t.InnerType == "VkFlags64" {
				baseType = "Flags64"
			}
			bitmasks = append(bitmasks, struct {
				name     string
				baseType string
			}{name, baseType})
		}
	}

	if len(bitmasks) > 0 {
		fmt.Fprintln(f, "// Bitmask types")
		fmt.Fprintln(f, "type (")
		for _, b := range bitmasks {
			goName := vkToGoType(b.name)
			fmt.Fprintf(f, "\t%s %s\n", goName, b.baseType)
		}
		fmt.Fprintln(f, ")")
		fmt.Fprintln(f, "")
	}

	// Platform-specific types as uintptr placeholders
	fmt.Fprintln(f, "// Platform-specific types (opaque pointers)")
	fmt.Fprintln(f, "type (")
	fmt.Fprintln(f, "\tANativeWindow          uintptr")
	fmt.Fprintln(f, "\tAHardwareBuffer        uintptr")
	fmt.Fprintln(f, "\tCAMetalLayer           uintptr")
	fmt.Fprintln(f, "\tWlDisplay              uintptr // wl_display")
	fmt.Fprintln(f, "\tWlSurface              uintptr // wl_surface")
	fmt.Fprintln(f, "\tXcbConnection          uintptr // xcb_connection_t")
	fmt.Fprintln(f, "\tXcbWindow              uint32  // xcb_window_t")
	fmt.Fprintln(f, "\tXcbVisualid            uint32  // xcb_visualid_t")
	fmt.Fprintln(f, "\tXlibDisplay            uintptr // Display*")
	fmt.Fprintln(f, "\tXlibWindow             uintptr // Window")
	fmt.Fprintln(f, "\tXlibVisualID           uintptr // VisualID")
	fmt.Fprintln(f, "\tZxBufferCollectionFUCHSIA uintptr")
	fmt.Fprintln(f, "\tGgpStreamDescriptor    uintptr")
	fmt.Fprintln(f, "\tGgpFrameToken          uintptr")
	fmt.Fprintln(f, "\tIDirectFB              uintptr")
	fmt.Fprintln(f, "\tIDirectFBSurface       uintptr")
	fmt.Fprintln(f, "\tScreenContext          uintptr // _screen_context")
	fmt.Fprintln(f, "\tScreenWindow           uintptr // _screen_window")
	fmt.Fprintln(f, "\tScreenBuffer           uintptr // _screen_buffer")
	fmt.Fprintln(f, "\tNvSciSyncAttrList      uintptr")
	fmt.Fprintln(f, "\tNvSciSyncObj           uintptr")
	fmt.Fprintln(f, "\tNvSciSyncFence         uintptr")
	fmt.Fprintln(f, "\tNvSciBufAttrList       uintptr")
	fmt.Fprintln(f, "\tNvSciBufObj            uintptr")
	fmt.Fprintln(f, "\tMTLDevice_id           uintptr")
	fmt.Fprintln(f, "\tMTLCommandQueue_id     uintptr")
	fmt.Fprintln(f, "\tMTLBuffer_id           uintptr")
	fmt.Fprintln(f, "\tMTLTexture_id          uintptr")
	fmt.Fprintln(f, "\tMTLSharedEvent_id      uintptr")
	fmt.Fprintln(f, "\tIOSurfaceRef           uintptr")
	fmt.Fprintln(f, ")")
	fmt.Fprintln(f, "")

	// Union types. The element type gives each the alignment of its widest
	// member, as in C: a byte array would be 1-aligned, so a struct holding
	// one would lay out its fields differently than the driver expects.
	fmt.Fprintln(f, "// Union types (largest member size, widest member's alignment)")
	fmt.Fprintln(f, "type ClearValue [4]uint32 // Union: ClearColorValue or ClearDepthStencilValue")
	fmt.Fprintln(f, "type ClearColorValue [4]uint32 // Union: float32[4], int32[4], or uint32[4]")
	fmt.Fprintln(f, "type PerformanceValueDataINTEL [1]uint64 // Union: value32, value64, valueFloat, valueBool, valueString")
	fmt.Fprintln(f, "type PipelineExecutableStatisticValueKHR [1]uint64 // Union: b32, i64, u64, f64")
	fmt.Fprintln(f, "type PerformanceCounterResultKHR [1]uint64 // Union: int32, int64, uint32, uint64, float32, float64")
	fmt.Fprintln(f, "type DeviceOrHostAddressKHR uint64 // Union: deviceAddress or hostAddress")
	fmt.Fprintln(f, "type DeviceOrHostAddressConstKHR uint64 // Union: deviceAddress or hostAddress")
	fmt.Fprintln(f, "type AccelerationStructureGeometryDataKHR AccelerationStructureGeometryTrianglesDataKHR // Union: triangles, aabbs, instances; triangles is the largest everywhere")
	fmt.Fprintln(f, "type AccelerationStructureMotionInstanceDataNV [18]uint64 // Union: static, matrix, srt instances")
	fmt.Fprintln(f, "type ClusterAccelerationStructureOpInputNV uintptr // Union: various cluster inputs")
	fmt.Fprintln(f, "type DescriptorDataEXT uint64 // Union: descriptor data variants, a VkDeviceAddress among them")
	fmt.Fprintln(f, "")

	// Extension type aliases
	fmt.Fprintln(f, "// Extension type aliases")
	fmt.Fprintln(f, "type AccelerationStructureTypeNV = AccelerationStructureTypeKHR")
	fmt.Fprintln(f, "type BuildAccelerationStructureFlagsNV = BuildAccelerationStructureFlagsKHR")
	fmt.Fprintln(f, "type ComponentTypeNV = ComponentTypeKHR")
	fmt.Fprintln(f, "type ScopeNV = ScopeKHR")
	fmt.Fprintln(f, "type GeometryTypeNV = GeometryTypeKHR")
	fmt.Fprintln(f, "type GeometryFlagsNV = GeometryFlagsKHR")
	fmt.Fprintln(f, "type GeometryInstanceFlagsNV = GeometryInstanceFlagsKHR")
	fmt.Fprintln(f, "type CopyAccelerationStructureModeNV = CopyAccelerationStructureModeKHR")
	fmt.Fprintln(f, "type RayTracingShaderGroupTypeNV = RayTracingShaderGroupTypeKHR")
	fmt.Fprintln(f, "")

	// More union types
	fmt.Fprintln(f, "// More union types")
	fmt.Fprintln(f, "type IndirectExecutionSetInfoEXT uintptr")
	fmt.Fprintln(f, "type IndirectCommandsTokenDataEXT uintptr")
	fmt.Fprintln(f, "")

	// Video codec types (external headers - placeholders)
	fmt.Fprintln(f, "// Video codec types (external from vulkan_video_codec_*.h)")
	fmt.Fprintln(f, "type (")
	fmt.Fprintln(f, "\tStdVideoH264ProfileIdc             int32")
	fmt.Fprintln(f, "\tStdVideoH264LevelIdc               int32")
	fmt.Fprintln(f, "\tStdVideoH264ChromaFormatIdc        int32")
	fmt.Fprintln(f, "\tStdVideoH264PocType                int32")
	fmt.Fprintln(f, "\tStdVideoH264AspectRatioIdc         int32")
	fmt.Fprintln(f, "\tStdVideoH264WeightedBipredIdc      int32")
	fmt.Fprintln(f, "\tStdVideoH264ModificationOfPicNumsIdc int32")
	fmt.Fprintln(f, "\tStdVideoH264MemMgmtControlOp       int32")
	fmt.Fprintln(f, "\tStdVideoH264CabacInitIdc           int32")
	fmt.Fprintln(f, "\tStdVideoH264DisableDeblockingFilterIdc int32")
	fmt.Fprintln(f, "\tStdVideoH264SliceType              int32")
	fmt.Fprintln(f, "\tStdVideoH264PictureType            int32")
	fmt.Fprintln(f, "\tStdVideoH264NonVclNaluType         int32")
	fmt.Fprintln(f, "\tStdVideoH264SequenceParameterSet   [256]byte")
	fmt.Fprintln(f, "\tStdVideoH264PictureParameterSet    [128]byte")
	fmt.Fprintln(f, "\tStdVideoDecodeH264PictureInfo      [64]byte")
	fmt.Fprintln(f, "\tStdVideoDecodeH264ReferenceInfo    [32]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH264PictureInfo      [64]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH264ReferenceInfo    [32]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH264SliceHeader      [128]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH264ReferenceListsInfo [64]byte")
	fmt.Fprintln(f, "\tStdVideoH265ProfileIdc             int32")
	fmt.Fprintln(f, "\tStdVideoH265LevelIdc               int32")
	fmt.Fprintln(f, "\tStdVideoH265ChromaFormatIdc        int32")
	fmt.Fprintln(f, "\tStdVideoH265AspectRatioIdc         int32")
	fmt.Fprintln(f, "\tStdVideoH265SliceType              int32")
	fmt.Fprintln(f, "\tStdVideoH265PictureType            int32")
	fmt.Fprintln(f, "\tStdVideoH265VideoParameterSet      [256]byte")
	fmt.Fprintln(f, "\tStdVideoH265SequenceParameterSet   [512]byte")
	fmt.Fprintln(f, "\tStdVideoH265PictureParameterSet    [256]byte")
	fmt.Fprintln(f, "\tStdVideoDecodeH265PictureInfo      [64]byte")
	fmt.Fprintln(f, "\tStdVideoDecodeH265ReferenceInfo    [32]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH265PictureInfo      [64]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH265ReferenceInfo    [32]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH265SliceSegmentHeader [128]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeH265ReferenceListsInfo [64]byte")
	fmt.Fprintln(f, "\tStdVideoAV1Profile                 int32")
	fmt.Fprintln(f, "\tStdVideoAV1Level                   int32")
	fmt.Fprintln(f, "\tStdVideoAV1SequenceHeader          [256]byte")
	fmt.Fprintln(f, "\tStdVideoDecodeAV1PictureInfo       [128]byte")
	fmt.Fprintln(f, "\tStdVideoDecodeAV1ReferenceInfo     [64]byte")
	fmt.Fprintln(f, "\tStdVideoAV1TileInfo                [64]byte")
	fmt.Fprintln(f, "\tStdVideoAV1Quantization            [32]byte")
	fmt.Fprintln(f, "\tStdVideoAV1Segmentation            [64]byte")
	fmt.Fprintln(f, "\tStdVideoAV1LoopFilter              [32]byte")
	fmt.Fprintln(f, "\tStdVideoAV1CDEF                    [32]byte")
	fmt.Fprintln(f, "\tStdVideoAV1LoopRestoration         [16]byte")
	fmt.Fprintln(f, "\tStdVideoAV1GlobalMotion            [64]byte")
	fmt.Fprintln(f, "\tStdVideoAV1FilmGrain               [128]byte")
	// VP9 codec types
	fmt.Fprintln(f, "\tStdVideoVP9Profile                 int32")
	fmt.Fprintln(f, "\tStdVideoVP9Level                   int32")
	fmt.Fprintln(f, "\tStdVideoDecodeVP9PictureInfo       [128]byte")
	// AV1 encode types
	fmt.Fprintln(f, "\tStdVideoEncodeAV1DecoderModelInfo  [32]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeAV1OperatingPointInfo [64]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeAV1ReferenceInfo     [32]byte")
	fmt.Fprintln(f, "\tStdVideoEncodeAV1PictureInfo       [64]byte")
	fmt.Fprintln(f, ")")
	fmt.Fprintln(f, "")

	// More extension union and bitmask types
	fmt.Fprintln(f, "// More extension types")
	fmt.Fprintln(f, "type MemoryDecompressionMethodFlagsNV = Flags64")
	fmt.Fprintln(f, "type DeviceOrHostAddressConstAMDX uint64")
	fmt.Fprintln(f, "type PipelineCreateFlags2KHR = Flags64")
	fmt.Fprintln(f, "")

	// OpenHarmony platform types
	fmt.Fprintln(f, "// OpenHarmony platform types")
	fmt.Fprintln(f, "type OHNativeWindow uintptr")
	fmt.Fprintln(f, "type OHBufferHandle uintptr")
	fmt.Fprintln(f, "type OH_NativeBuffer uintptr")
	fmt.Fprintln(f, "")

	// Handle types. A dispatchable handle is a pointer, a non-dispatchable
	// one 64 bits wide on every target: vulkan_core.h makes it a uint64_t
	// where pointers are 32 bits.
	handles := []string{}
	dispatchable := map[string]bool{}
	aliases := map[string]string{}
	for _, t := range registry.Types.Types {
		if t.Category == "handle" {
			name := t.Name
			if name == "" {
				name = t.InnerName
			}
			if name != "" && !strings.HasPrefix(name, "Vk") {
				continue
			}
			handles = append(handles, name)
			if t.Alias != "" {
				aliases[name] = t.Alias
			} else {
				dispatchable[name] = t.InnerType == "VK_DEFINE_HANDLE"
			}
		}
	}

	if len(handles) > 0 {
		fmt.Fprintln(f, "// Handles")
		fmt.Fprintln(f, "type (")
		for _, h := range handles {
			goName := vkToGoType(h)
			base := h
			if a, ok := aliases[h]; ok {
				base = a
			}
			goType := "uint64"
			if dispatchable[base] {
				goType = "uintptr"
			}
			fmt.Fprintf(f, "\t%s %s\n", goName, goType)
		}
		fmt.Fprintln(f, ")")
		fmt.Fprintln(f, "")
	}

	// Structs
	seenTypes := make(map[string]bool)
	// Skip types that are unions (defined manually above)
	skipTypes := map[string]bool{
		"ClearValue":                                true, // Union type - defined above
		"ClearColorValue":                           true, // Union type - defined above
		"PerformanceValueDataINTEL":                 true,
		"PipelineExecutableStatisticValueKHR":       true,
		"PerformanceCounterResultKHR":               true,
		"DeviceOrHostAddressKHR":                    true,
		"DeviceOrHostAddressConstKHR":               true,
		"AccelerationStructureGeometryDataKHR":      true,
		"AccelerationStructureMotionInstanceDataNV": true,
		"ClusterAccelerationStructureOpInputNV":     true,
		"DescriptorDataEXT":                         true,
		"IndirectExecutionSetInfoEXT":               true,
		"IndirectCommandsTokenDataEXT":              true,
	}
	// Structs that C lays out with padding Go leaves out where 64-bit
	// members are aligned to 8 bytes and pointers are 4 bytes wide get a
	// definition with that padding there, and the plain one elsewhere.
	var align4, align8 bytes.Buffer
	for _, t := range registry.Types.Types {
		if t.Category != "struct" || t.Alias != "" {
			continue
		}
		if len(t.Members) == 0 {
			continue
		}

		name := t.Name
		goName := vkToGoType(name)

		if seenTypes[goName] || skipTypes[goName] {
			continue
		}
		seenTypes[goName] = true

		var fields [][2]string
		seenMembers := make(map[string]bool)
		for _, m := range t.Members {
			memberName := goFieldName(m.Name)
			// Skip duplicate members
			if seenMembers[memberName] {
				continue
			}
			seenMembers[memberName] = true
			fields = append(fields, [2]string{memberName, vkToGoFieldType(m.Type, m.RawXML, m.Enum)})
		}
		pad, padded := pads[goName]
		if !padded {
			writeStruct(f, name, goName, fields, nil)
		} else {
			writeStruct(&align4, name, goName, fields, nil)
			writeStruct(&align8, name, goName, fields, pad)
		}
	}

	for _, file := range []struct {
		name, constraint string
		body             *bytes.Buffer
	}{
		{"types_align4_gen.go", "!(arm || mips || mipsle || (windows && 386))", &align4},
		{"types_align8_gen.go", "arm || mips || mipsle || (windows && 386)", &align8},
	} {
		path := filepath.Join(outDir, file.name)
		if pads == nil {
			os.Remove(path)
			continue
		}
		var b bytes.Buffer
		fmt.Fprintf(&b, "//go:build %s\n\n", file.constraint)
		fmt.Fprintln(&b, "// Code generated by vk-gen. DO NOT EDIT.")
		fmt.Fprintln(&b, "")
		fmt.Fprintln(&b, "package vk")
		fmt.Fprintln(&b, "")
		fmt.Fprintln(&b, "// The structs C lays out with padding ahead of a 64-bit member, or after")
		fmt.Fprintln(&b, "// the last, on 32-bit targets that align 64-bit members to 8 bytes: arm,")
		fmt.Fprintln(&b, "// mips and Windows on 386. Go aligns them to 4 there, so types_align8_gen.go")
		fmt.Fprintln(&b, "// spells the padding out, and types_align4_gen.go has them as they are for")
		fmt.Fprintln(&b, "// the targets Go lays them out like C on.")
		fmt.Fprintln(&b, "")
		b.Write(file.body.Bytes())
		if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
			return err
		}
	}
	return nil
}

// writeStruct writes a struct definition with its fields, and pad[i] bytes
// of padding ahead of field i and pad[len(fields)] after the last, if pad is
// set.
func writeStruct(w io.Writer, vkName, goName string, fields [][2]string, pad []int64) {
	fmt.Fprintf(w, "// %s\n", vkName)
	fmt.Fprintf(w, "type %s struct {\n", goName)
	for i, field := range fields {
		if pad != nil && pad[i] > 0 {
			fmt.Fprintf(w, "\t_ [%d]byte\n", pad[i])
		}
		fmt.Fprintf(w, "\t%s %s\n", field[0], field[1])
	}
	if pad != nil && pad[len(fields)] > 0 {
		fmt.Fprintf(w, "\t_ [%d]byte\n", pad[len(fields)])
	}
	fmt.Fprintln(w, "}")
	fmt.Fprintln(w, "")
}

func generateCommands(registry *Registry, outDir string) error {
	f, err := os.Create(filepath.Join(outDir, "commands_gen.go"))
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Fprintln(f, "//go:build !(js && wasm)")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "// Code generated by vk-gen. DO NOT EDIT.")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "package vk")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "// FFI error handling follows ADR-049 three-tier strategy:")
	fmt.Fprintln(f, "//   Tier 1 (creation/submit): check both FFI error and API result code")
	fmt.Fprintln(f, "//   Tier 2 (void/hot path): infallible by GPU API contract — Vulkan §6.6, WebGPU §21.2")
	fmt.Fprintln(f, "//   Tier 3 (platform syscalls): use errno for diagnostics (Wayland, X11, Win32)")
	fmt.Fprintln(f, "// Enterprise reference: Rust wgpu-hal returns () for all draw/destroy/barrier commands.")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "import (")
	fmt.Fprintln(f, "\t\"unsafe\"")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "\t\"github.com/go-webgpu/goffi/ffi\"")
	fmt.Fprintln(f, ")")
	fmt.Fprintln(f, "")
	fmt.Fprintln(f, "// Prevent unused import error")
	fmt.Fprintln(f, "var _ = unsafe.Sizeof(0)")
	fmt.Fprintln(f, "")

	// Generate command function pointer struct
	fmt.Fprintln(f, "// Commands holds Vulkan function pointers.")
	fmt.Fprintln(f, "type Commands struct {")

	seen := make(map[string]bool)
	commands := make([]Command, 0, len(registry.Commands.Commands))

	for _, cmd := range registry.Commands.Commands {
		if cmd.Alias != "" {
			continue
		}
		name := cmd.Proto.Name
		if name == "" {
			name = cmd.Name
		}
		if name == "" {
			continue
		}
		goName := strings.TrimPrefix(name, "vk")
		goName = strings.ToLower(goName[:1]) + goName[1:]
		if seen[goName] {
			continue
		}
		seen[goName] = true
		fmt.Fprintf(f, "\t%s unsafe.Pointer\n", goName)
		commands = append(commands, cmd)
	}
	fmt.Fprintln(f, "}")
	fmt.Fprintln(f, "")

	// Generate wrapper methods for each command
	for _, cmd := range commands {
		if err := generateCommandMethod(f, cmd); err != nil {
			fmt.Printf("Warning: skipping command %s: %v\n", cmd.Proto.Name, err)
		}
	}

	return nil
}

// filterParams filters out vulkansc-only parameters and deduplicates by name.
// Vulkan XML can have multiple params with the same name but different api attributes:
//   - api="vulkan,vulkanbase" - standard Vulkan
//   - api="vulkansc" - Vulkan Safety Critical (not supported)
//
// We only want standard Vulkan params, and we need to handle cases where
// the same param appears multiple times with different api values.
func filterParams(params []Param) []Param {
	seen := make(map[string]bool)
	result := make([]Param, 0, len(params))

	for _, p := range params {
		// Skip vulkansc-only params
		if p.API == "vulkansc" {
			continue
		}

		// Skip duplicates (take first occurrence for each name)
		if seen[p.Name] {
			continue
		}
		seen[p.Name] = true
		result = append(result, p)
	}

	return result
}

// isVulkanEnum returns true if the type is a known Vulkan enum type.
// This is a conservative list - only known enums are included.
// Handles like VkBuffer, VkImage, VkPipeline should NOT be classified as enums.
func isVulkanEnum(typeName string) bool {
	// Known enum types (explicit list to avoid false positives on handles)
	knownEnums := map[string]bool{
		// Core enums
		"VkResult":                            true,
		"VkFormat":                            true,
		"VkImageLayout":                       true,
		"VkImageType":                         true,
		"VkImageTiling":                       true,
		"VkImageViewType":                     true,
		"VkSampleCountFlagBits":               true,
		"VkPipelineBindPoint":                 true,
		"VkIndexType":                         true,
		"VkCommandBufferLevel":                true,
		"VkPipelineStageFlagBits":             true,
		"VkSharingMode":                       true,
		"VkComponentSwizzle":                  true,
		"VkBorderColor":                       true,
		"VkFilter":                            true,
		"VkSamplerMipmapMode":                 true,
		"VkSamplerAddressMode":                true,
		"VkCompareOp":                         true,
		"VkStencilOp":                         true,
		"VkLogicOp":                           true,
		"VkBlendFactor":                       true,
		"VkBlendOp":                           true,
		"VkDynamicState":                      true,
		"VkPrimitiveTopology":                 true,
		"VkPolygonMode":                       true,
		"VkCullModeFlagBits":                  true,
		"VkFrontFace":                         true,
		"VkVertexInputRate":                   true,
		"VkAttachmentLoadOp":                  true,
		"VkAttachmentStoreOp":                 true,
		"VkDescriptorType":                    true,
		"VkQueryType":                         true,
		"VkSubpassContents":                   true,
		"VkPresentModeKHR":                    true,
		"VkColorSpaceKHR":                     true,
		"VkObjectType":                        true,
		"VkDebugReportObjectTypeEXT":          true,
		"VkQueryResultFlagBits":               true,
		"VkBufferUsageFlagBits":               true,
		"VkMemoryPropertyFlagBits":            true,
		"VkQueueFlagBits":                     true,
		"VkImageUsageFlagBits":                true,
		"VkImageAspectFlagBits":               true,
		"VkAccessFlagBits":                    true,
		"VkDependencyFlagBits":                true,
		"VkPipelineCreateFlagBits":            true,
		"VkShaderStageFlagBits":               true,
		"VkCommandPoolCreateFlagBits":         true,
		"VkCommandPoolResetFlagBits":          true,
		"VkCommandBufferUsageFlagBits":        true,
		"VkCommandBufferResetFlagBits":        true,
		"VkFenceCreateFlagBits":               true,
		"VkSemaphoreCreateFlagBits":           true,
		"VkEventCreateFlagBits":               true,
		"VkQueryPipelineStatisticFlagBits":    true,
		"VkImageCreateFlagBits":               true,
		"VkSamplerCreateFlagBits":             true,
		"VkDescriptorPoolCreateFlagBits":      true,
		"VkDescriptorSetLayoutCreateFlagBits": true,
		"VkPipelineLayoutCreateFlagBits":      true,
		"VkRenderPassCreateFlagBits":          true,
		"VkSubpassDescriptionFlagBits":        true,
		"VkFramebufferCreateFlagBits":         true,
		"VkColorComponentFlagBits":            true,
		"VkStencilFaceFlagBits":               true,
		"VkCullModeFlags":                     true,
		"VkSurfaceTransformFlagBitsKHR":       true,
		"VkCompositeAlphaFlagBitsKHR":         true,
		"VkSwapchainCreateFlagBitsKHR":        true,
		"VkDisplayPlaneAlphaFlagBitsKHR":      true,
		"VkDebugReportFlagBitsEXT":            true,
	}

	return knownEnums[typeName]
}

// generateCommandMethod generates a wrapper method for a Vulkan command
func generateCommandMethod(f *os.File, cmd Command) error {
	name := cmd.Proto.Name
	if name == "" {
		name = cmd.Name
	}

	// Convert vkFunctionName to FunctionName (exported Go method)
	goMethodName := strings.TrimPrefix(name, "vk")

	// Convert vkFunctionName to functionName (struct field)
	goFieldName := strings.TrimPrefix(name, "vk")
	goFieldName = strings.ToLower(goFieldName[:1]) + goFieldName[1:]

	// The wrapper is written in goffi's form, and vk-cgo turns it into a call
	// through the C trampoline for its signature, which it reads off the Go
	// parameter types. So every signature works, and the CallFunction
	// interface named here is only a placeholder vk-cgo drops.
	const callInterface = "cgoTrampoline"

	// Filter params (exclude vulkansc, deduplicate)
	params := filterParams(cmd.Params)

	// Generate method signature
	fmt.Fprintf(f, "// %s wraps %s.\n", goMethodName, name)
	fmt.Fprintf(f, "func (c *Commands) %s(", goMethodName)

	// Parameters
	for i, param := range params {
		if i > 0 {
			fmt.Fprint(f, ", ")
		}
		paramName := goParamName(param.Name)
		paramType := vkToGoParamType(param)
		fmt.Fprintf(f, "%s %s", paramName, paramType)
	}

	// Return type
	returnType := cmd.Proto.Type
	if returnType == "VkResult" {
		fmt.Fprint(f, ") Result")
	} else if returnType != "void" {
		// Handle other return types if needed
		goReturnType := vkToGoBaseType(returnType)
		fmt.Fprintf(f, ") %s", goReturnType)
	} else {
		fmt.Fprint(f, ")")
	}

	fmt.Fprintln(f, " {")

	// vk-cgo reads these to hand C an array argument as a slice of its
	// length, whose every element cgocheck then walks.
	for _, param := range params {
		if count := countParam(param, params); count != "" {
			fmt.Fprintf(f, "\t//vk:len %s %s\n", goParamName(param.Name), goParamName(count))
		}
	}

	// Method body
	hasReturn := returnType == "VkResult" || returnType != "void"
	if hasReturn {
		if returnType == "VkResult" {
			fmt.Fprintln(f, "\tvar result int32")
		} else {
			goReturnType := vkToGoBaseType(returnType)
			fmt.Fprintf(f, "\tvar result %s\n", goReturnType)
		}
	}

	// Build args array.
	// goffi convention (ADR-044): args[i] = unsafe.Pointer to VALUE.
	// For pointer args: args[i] = unsafe.Pointer(&ptr) — pointer to pointer.
	// C array params (e.g., const float[4]) decay to pointers in C ABI,
	// so they need the same double-pointer treatment.
	argCount := len(params)

	// Emit local pointer vars for array params (C arrays decay to pointers).
	for _, param := range params {
		if strings.Contains(param.RawXML, "[") && !strings.Contains(param.RawXML, "*") {
			fmt.Fprintf(f, "\t%sPtr := unsafe.Pointer(&%s)\n", goParamName(param.Name), goParamName(param.Name))
		}
	}

	fmt.Fprintf(f, "\targs := [%d]unsafe.Pointer{\n", argCount)
	for _, param := range params {
		paramName := goParamName(param.Name)
		if strings.Contains(param.RawXML, "[") && !strings.Contains(param.RawXML, "*") {
			// Array param: pass pointer-to-pointer (C array decays to pointer)
			fmt.Fprintf(f, "\t\tunsafe.Pointer(&%sPtr),\n", paramName)
		} else {
			fmt.Fprintf(f, "\t\tunsafe.Pointer(&%s),\n", paramName)
		}
	}
	fmt.Fprintln(f, "\t}")

	// Call ffi.CallFunction with goffi error checking (VK-002).
	// goffi v0.6.0 returns (syscall.Errno, error); errno unused for Vulkan (VkResult is the API error).
	// goffi returns zeros on nil function pointer — no crash, no error —
	// which produces false VK_SUCCESS with null output handles (gogpu#119).
	resultPtr := "nil"
	if hasReturn {
		resultPtr = "unsafe.Pointer(&result)"
	}
	if returnType == "VkResult" {
		// Tier 1 (ADR-049): VkResult functions — check FFI transport error, then API result.
		fmt.Fprintf(f, "\tif _, err := ffi.CallFunction(&%s, c.%s, %s, args[:]); err != nil {\n", callInterface, goFieldName, resultPtr)
		fmt.Fprintln(f, "\t\treturn ErrorInitializationFailed")
		fmt.Fprintln(f, "\t}")
	} else {
		// Tier 2 (ADR-049): void/query functions — infallible by Vulkan spec §6.6.
		fmt.Fprintf(f, "\tif c.%s == nil {\n\t\treturn", goFieldName)
		if hasReturn && returnType != "VkResult" {
			fmt.Fprint(f, " 0")
		}
		fmt.Fprintln(f, "\n\t}")
		fmt.Fprintf(f, "\t_, _ = ffi.CallFunction(&%s, c.%s, %s, args[:])\n", callInterface, goFieldName, resultPtr)
	}

	// Return result
	if hasReturn {
		if returnType == "VkResult" {
			fmt.Fprintln(f, "\treturn Result(result)")
		} else {
			fmt.Fprintln(f, "\treturn result")
		}
	}

	fmt.Fprintln(f, "}")
	fmt.Fprintln(f, "")

	return nil
}

// countParam is the parameter passed by value that counts the elements
// param points at, from param's len, or "" when there is none.
func countParam(param Param, params []Param) string {
	if !strings.Contains(param.RawXML, "*") {
		return ""
	}
	for _, p := range params {
		if p.Name == param.Len && !strings.Contains(p.RawXML, "*") {
			return p.Name
		}
	}
	return ""
}

// goParamName keeps a Vulkan parameter name from being a Go keyword: three
// image format queries have a parameter called type.
func goParamName(name string) string {
	switch {
	case name == "type":
		return "typ"
	case token.IsKeyword(name):
		return name + "_"
	}
	return name
}

// vkToGoParamType converts Vulkan parameter type to Go type
func vkToGoParamType(param Param) string {
	return vkToGoFieldType(param.Type, param.RawXML, "")
}

func generateLoader(_ *Registry, outDir string) error {
	// NOTE: We do NOT generate loader_gen.go anymore!
	// The manual loader.go file in hal/vulkan/vk/loader.go already handles:
	// - Cross-platform library loading via goffi
	// - GetInstanceProcAddr and GetDeviceProcAddr wrappers
	// - InitSignatures() call
	//
	// Users will manually implement LoadGlobal/LoadInstance/LoadDevice methods
	// in Commands struct, or we can add a separate generator for that later.

	fmt.Println("  Skipping loader_gen.go generation (using manual loader.go)")
	return nil
}

// convertCValue converts C-style constant values to Go
func convertCValue(value string) string {
	// Handle C-style suffixes and expressions
	value = strings.TrimSpace(value)

	// Float with F suffix: 1000.0F -> 1000.0
	if strings.HasSuffix(value, "F") || strings.HasSuffix(value, "f") {
		return strings.TrimSuffix(strings.TrimSuffix(value, "F"), "f")
	}

	// Unsigned long long: (~0ULL) -> ^uint64(0)
	if strings.Contains(value, "ULL") {
		value = strings.ReplaceAll(value, "(~0ULL)", "^uint64(0)")
		value = strings.ReplaceAll(value, "~0ULL", "^uint64(0)")
		value = strings.ReplaceAll(value, "ULL", "")
		return value
	}

	// Unsigned: (~0U), (~1U), (~2U) -> ^uint32(0), etc.
	if strings.Contains(value, "U)") || strings.HasSuffix(value, "U") {
		// (~0U) -> ^uint32(0)
		if strings.HasPrefix(value, "(~") && strings.HasSuffix(value, "U)") {
			inner := strings.TrimPrefix(value, "(~")
			inner = strings.TrimSuffix(inner, "U)")
			return "^uint32(" + inner + ")"
		}
		// Simple U suffix
		value = strings.TrimSuffix(value, "U")
		return value
	}

	return value
}

// Constants that conflict with type names - add Value suffix
var conflictingConstants = map[string]bool{
	"PipelineCacheHeaderVersionOne": true,
}

// vkToGoConst converts VK_CONSTANT_NAME to GoConstantName
func vkToGoConst(name string) string {
	// VK_SUCCESS -> Success
	// VK_ERROR_OUT_OF_HOST_MEMORY -> ErrorOutOfHostMemory
	name = strings.TrimPrefix(name, "VK_")
	parts := strings.Split(name, "_")
	var result strings.Builder
	for _, p := range parts {
		if p == "" {
			continue
		}
		result.WriteString(strings.ToUpper(p[:1]))
		result.WriteString(strings.ToLower(p[1:]))
	}
	goName := result.String()
	// Check for conflicts with type names
	if conflictingConstants[goName] {
		return goName + "Value"
	}
	return goName
}

// vkToGoType converts VkTypeName to TypeName
func vkToGoType(name string) string {
	return strings.TrimPrefix(name, "Vk")
}

// goFieldName converts sType to SType (exported Go field name)
func goFieldName(name string) string {
	if name == "" {
		return ""
	}
	// Handle special cases
	switch name {
	case "sType":
		return "SType"
	case "pNext":
		return "PNext"
	}
	// Capitalize first letter
	return strings.ToUpper(name[:1]) + name[1:]
}

// vkToGoFieldType converts Vulkan type to Go type
func vkToGoFieldType(vkType string, rawXML string, enumSize string) string {
	// Check for pointer
	isPointer := strings.Contains(rawXML, "*")
	isDoublePointer := strings.Contains(rawXML, "**") || strings.Contains(rawXML, "* const*")

	// Check for array: [N] or [ENUM_CONSTANT]
	isArray := strings.Contains(rawXML, "[")

	baseType := vkToGoBaseType(vkType)

	// Pointers stay typed Go pointers all the way to the C call: a pointer
	// hidden in a uintptr is neither kept alive nor fixed up when a stack
	// copy moves what it points at.
	if isDoublePointer {
		switch {
		case vkType == "char":
			return "**byte" // const char* const*: string arrays
		case vkType == "void":
			return "*unsafe.Pointer" // void**: mapped memory and such
		}
		return "**" + baseType
	}
	if isPointer {
		if baseType == "byte" || vkType == "char" {
			return "*byte" // NUL-terminated string
		}
		return "*" + baseType
	}

	// Handle arrays
	if isArray {
		// Extract array size from rawXML like "[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]"
		// or from enumSize field
		arraySize := ""
		if enumSize != "" {
			// Convert VK_MAX_PHYSICAL_DEVICE_NAME_SIZE to the constant value
			arraySize = convertEnumToSize(enumSize)
		} else {
			// Try to extract from rawXML
			start := strings.Index(rawXML, "[")
			end := strings.Index(rawXML, "]")
			if start >= 0 && end > start {
				sizeStr := rawXML[start+1 : end]
				// Remove any nested tags like <enum>
				sizeStr = strings.TrimPrefix(sizeStr, "<enum>")
				sizeStr = strings.TrimSuffix(sizeStr, "</enum>")
				arraySize = convertEnumToSize(sizeStr)
			}
		}
		if arraySize != "" {
			return "[" + arraySize + "]" + baseType
		}
	}

	return baseType
}

// convertEnumToSize converts VK_* constants to their numeric values
func convertEnumToSize(enumName string) string {
	// Map of known array size constants
	sizeMap := map[string]string{
		"VK_MAX_PHYSICAL_DEVICE_NAME_SIZE":          "256",
		"VK_UUID_SIZE":                              "16",
		"VK_LUID_SIZE":                              "8",
		"VK_MAX_EXTENSION_NAME_SIZE":                "256",
		"VK_MAX_DESCRIPTION_SIZE":                   "256",
		"VK_MAX_MEMORY_TYPES":                       "32",
		"VK_MAX_MEMORY_HEAPS":                       "16",
		"VK_MAX_DRIVER_NAME_SIZE":                   "256",
		"VK_MAX_DRIVER_INFO_SIZE":                   "256",
		"VK_MAX_DEVICE_GROUP_SIZE":                  "32",
		"VK_MAX_GLOBAL_PRIORITY_SIZE_KHR":           "16",
		"VK_MAX_GLOBAL_PRIORITY_SIZE_EXT":           "16",
		"VK_MAX_SHADER_MODULE_IDENTIFIER_SIZE_EXT":  "32",
		"VK_MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR": "7",
		// Matrix sizes
		"3":  "3",
		"4":  "4",
		"2":  "2",
		"12": "12",
	}

	if size, ok := sizeMap[enumName]; ok {
		return size
	}

	// If it's a plain number, return as is
	if _, err := strconv.Atoi(enumName); err == nil {
		return enumName
	}

	// Unknown constant - use a default or return empty
	return ""
}

func vkToGoBaseType(vkType string) string {
	switch vkType {
	case "void":
		return "uintptr"
	case "char":
		return "byte"
	case "float":
		return "float32"
	case "double":
		return "float64"
	case "int8_t":
		return "int8"
	case "uint8_t":
		return "uint8"
	case "int16_t":
		return "int16"
	case "uint16_t":
		return "uint16"
	case "int32_t":
		return "int32"
	case "int": // file descriptors and gralloc values: 32 bits wherever Vulkan runs
		return "int32"
	case "uint32_t":
		return "uint32"
	case "int64_t":
		return "int64"
	case "uint64_t":
		return "uint64"
	case "size_t":
		return "uintptr"
	case "VkBool32":
		return "Bool32"
	case "VkDeviceSize":
		return "DeviceSize"
	case "VkDeviceAddress":
		return "DeviceAddress"
	case "VkFlags":
		return "Flags"
	case "VkFlags64":
		return "Flags64"
	case "VkSampleMask":
		return "SampleMask"
	// Platform-specific types
	case "ANativeWindow":
		return "ANativeWindow"
	case "AHardwareBuffer":
		return "AHardwareBuffer"
	case "CAMetalLayer":
		return "CAMetalLayer"
	case "wl_display":
		return "WlDisplay"
	case "wl_surface":
		return "WlSurface"
	case "xcb_connection_t":
		return "XcbConnection"
	case "xcb_window_t":
		return "XcbWindow"
	case "xcb_visualid_t":
		return "XcbVisualid"
	case "Display":
		return "XlibDisplay"
	case "Window":
		return "XlibWindow"
	case "VisualID":
		return "XlibVisualID"
	case "zx_handle_t":
		return "uint32"
	case "GgpStreamDescriptor":
		return "GgpStreamDescriptor"
	case "GgpFrameToken":
		return "GgpFrameToken"
	case "IDirectFB":
		return "IDirectFB"
	case "IDirectFBSurface":
		return "IDirectFBSurface"
	case "_screen_context":
		return "ScreenContext"
	case "_screen_window":
		return "ScreenWindow"
	case "_screen_buffer":
		return "ScreenBuffer"
	case "NvSciSyncAttrList":
		return "NvSciSyncAttrList"
	case "NvSciSyncObj":
		return "NvSciSyncObj"
	case "NvSciSyncFence":
		return "NvSciSyncFence"
	case "NvSciBufAttrList":
		return "NvSciBufAttrList"
	case "NvSciBufObj":
		return "NvSciBufObj"
	case "MTLDevice_id":
		return "MTLDevice_id"
	case "MTLCommandQueue_id":
		return "MTLCommandQueue_id"
	case "MTLBuffer_id":
		return "MTLBuffer_id"
	case "MTLTexture_id":
		return "MTLTexture_id"
	case "MTLSharedEvent_id":
		return "MTLSharedEvent_id"
	case "IOSurfaceRef":
		return "IOSurfaceRef"
	case "HINSTANCE":
		return "uintptr"
	case "HWND":
		return "uintptr"
	case "HMONITOR":
		return "uintptr"
	case "HANDLE":
		return "uintptr"
	case "DWORD":
		return "uint32"
	case "LPCWSTR":
		return "uintptr"
	case "SECURITY_ATTRIBUTES":
		return "uintptr"
	case "RROutput":
		return "uintptr"
	default:
		if strings.HasPrefix(vkType, "Vk") {
			return strings.TrimPrefix(vkType, "Vk")
		}
		if strings.HasPrefix(vkType, "PFN_") {
			return "uintptr" // Function pointers
		}
		return vkType
	}
}
