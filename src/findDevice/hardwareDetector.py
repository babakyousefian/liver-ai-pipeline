# src/findDevice/hardwareDetector.py
# -*- coding: utf-8 -*-

"""
Hardware Detection Module (v1.0)

Comprehensive hardware detection for the Liver AI Pipeline.
Extracts detailed specs from the local system using WMI, psutil, platform, and winreg.

Extracted Information:
- System: Brand, Model, Chassis
- CPU: Name, Code Name, TDP, Package, Technology, VID, Specification,
       Cache (L1/L2/L3), Cores, Threads, Max Clock, Current Clock
- Motherboard: Manufacturer, Model, Bus Specs, Chipset, South Bridge
- BIOS: Brand, Version, Release Date, SMBIOS Version, ROM Size
- Memory: Total RAM, Type (DDR3/4/5), Speed, Manufacturer, Slot Info, ROM
- Graphics: All GPUs with name, VRAM, type (integrated/dedicated)

Beautiful terminal output using Rich (fallback to ANSI colors).

Author: Babak Yousefian
Version: 1.0.0
"""

import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

# ================================================================
# ✅ Try imports
# ================================================================

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not installed. Install: pip install psutil")

try:
    import wmi
    HAS_WMI = True
except ImportError:
    HAS_WMI = False
    print("⚠️ wmi not installed. Install: pip install wmi pywin32")

try:
    import winreg
    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Rich
try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.align import Align
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


# ================================================================
# ✅ Color Palette
# ================================================================

class Colors:
    """ANSI color codes for fallback display."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


# ================================================================
# ✅ WMI Helpers
# ================================================================

def _wmi_query(wmi_conn, class_name: str, fields: Optional[List[str]] = None) -> List[Dict]:
    """
    Safe WMI query. Returns list of dicts.

    Args:
        wmi_conn: WMI connection object
        class_name: e.g. "Win32_Processor"
        fields: list of fields to extract (None = all)

    Returns:
        List of dicts.
    """
    results = []
    try:
        items = wmi_conn.query(f"SELECT * FROM {class_name}")
        for item in items:
            if fields:
                row = {}
                for f in fields:
                    try:
                        row[f] = getattr(item, f, None)
                    except Exception:
                        row[f] = None
                results.append(row)
            else:
                results.append(item)
    except Exception as e:
        pass
    return results


def _safe_get(obj, attr: str, default=None):
    """Safely get attribute from WMI object."""
    try:
        val = getattr(obj, attr, default)
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default


def _format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable (GB, MB)."""
    if size_bytes is None:
        return "N/A"
    try:
        size_bytes = int(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    except Exception:
        return "N/A"


def _format_mhz(mhz) -> str:
    """Format MHz to GHz."""
    if mhz is None:
        return "N/A"
    try:
        mhz = int(mhz)
        if mhz >= 1000:
            return f"{mhz/1000:.2f} GHz ({mhz} MHz)"
        return f"{mhz} MHz"
    except Exception:
        return "N/A"


def _guess_cpu_tdp(cpu_name: str) -> str:
    """
    Guess CPU TDP from name pattern.
    Note: TDP is NOT exposed via WMI. This is a heuristic.
    """
    cpu_lower = cpu_name.lower()
    # Intel patterns
    if "i9" in cpu_lower:
        if "hx" in cpu_lower or "hk" in cpu_lower:
            return "45-55W (HX/HK series)"
        return "35-65W (depends on model)"
    if "i7" in cpu_lower:
        if "hx" in cpu_lower or "hk" in cpu_lower:
            return "45-55W (HX/HK series)"
        if "h" in cpu_lower:
            return "35-45W (H series)"
        if "u" in cpu_lower:
            return "12-28W (U series)"
        if "p" in cpu_lower:
            return "28W (P series)"
        return "35-65W (depends on model)"
    if "i5" in cpu_lower:
        if "h" in cpu_lower:
            return "35-45W (H series)"
        if "u" in cpu_lower:
            return "12-28W (U series)"
        if "p" in cpu_lower:
            return "28W (P series)"
        return "35-65W (depends on model)"
    if "i3" in cpu_lower:
        return "15-35W (depends on model)"
    if "ryzen" in cpu_lower:
        if "9" in cpu_lower:
            return "45-105W (depends on model)"
        if "7" in cpu_lower:
            return "35-105W"
        if "5" in cpu_lower:
            return "35-95W"
        return "15-65W"
    if "xeon" in cpu_lower:
        return "85-165W"
    if "celeron" in cpu_lower or "pentium" in cpu_lower:
        return "6-15W"
    return "N/A (TDP not exposed via WMI)"


# ================================================================
# ✅ Detection Functions
# ================================================================

def get_system_info(wmi_conn) -> Dict[str, str]:
    """Get basic system information (brand, model, chassis)."""
    info = {
        "manufacturer": "Unknown",
        "model": "Unknown",
        "system_type": "Unknown",
        "serial": "N/A",
        "chassis": "Unknown",
        "os_name": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "os_arch": platform.machine(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }

    if wmi_conn:
        # Win32_ComputerSystem
        cs_list = _wmi_query(wmi_conn, "Win32_ComputerSystem")
        if cs_list:
            cs = cs_list[0]
            info["manufacturer"] = _safe_get(cs, "Manufacturer", "Unknown")
            info["model"] = _safe_get(cs, "Model", "Unknown")
            info["system_type"] = _safe_get(cs, "SystemType", "Unknown")

        # Win32_SystemEnclosure (chassis type)
        enc_list = _wmi_query(wmi_conn, "Win32_SystemEnclosure")
        if enc_list:
            enc = enc_list[0]
            chassis_types = _safe_get(enc, "ChassisTypes", None)
            chassis_map = {
                1: "Other", 2: "Unknown", 3: "Desktop", 4: "Low Profile Desktop",
                5: "Pizza Box", 6: "Mini Tower", 7: "Tower", 8: "Portable",
                9: "Laptop", 10: "Notebook", 11: "Hand Held", 12: "Docking Station",
                13: "All in One", 14: "Sub Notebook", 15: "Space-Saving",
                16: "Lunch Box", 17: "Main System Chassis", 18: "Expansion Chassis",
                19: "Sub Chassis", 20: "Bus Expansion Chassis", 21: "Peripheral Chassis",
                22: "Storage Chassis", 23: "Rack Mount Chassis", 24: "Sealed-Case PC",
            }
            if chassis_types:
                try:
                    # chassis_types is a tuple of ints
                    ct = int(chassis_types[0]) if isinstance(chassis_types, (list, tuple)) else int(chassis_types)
                    info["chassis"] = chassis_map.get(ct, f"Unknown ({ct})")
                except Exception:
                    pass

        # Win32_BIOS for serial
        bios_list = _wmi_query(wmi_conn, "Win32_BIOS")
        if bios_list:
            bios = bios_list[0]
            info["serial"] = _safe_get(bios, "SerialNumber", "N/A")

    # Simplify manufacturer name
    mfr = info["manufacturer"]
    brand_map = {
        "LENOVO": "Lenovo", "DELL INC.": "Dell", "DELL": "Dell",
        "ACER": "Acer", "ASUSTEK COMPUTER INC.": "ASUS", "ASUS": "ASUS",
        "HEWLETT-PACKARD": "HP", "HP": "HP", "HPE": "HPE",
        "MSI": "MSI", "MICRO-STAR INTERNATIONAL CO.,LTD": "MSI",
        "SAMSUNG ELECTRONICS CO., LTD.": "Samsung",
        "TOSHIBA": "Toshiba", "FUJITSU": "Fujitsu",
        "SONY": "Sony", "LG ELECTRONICS": "LG",
        "APPLE INC.": "Apple", "APPLE": "Apple",
        "RAZER": "Razer", "GIGABYTE": "Gigabyte",
        "MICROSOFT CORPORATION": "Microsoft",
        "INTEL CORPORATION": "Intel",
        "CLEVO": "Clevo", "XIAOMI": "Xiaomi",
    }
    info["manufacturer_display"] = brand_map.get(mfr.upper(), mfr)

    return info


def get_cpu_info(wmi_conn) -> List[Dict[str, str]]:
    """Get detailed CPU info. Returns list (support multi-CPU)."""
    cpus = []

    if not wmi_conn:
        # Fallback to platform
        return [{
            "name": platform.processor() or "Unknown",
            "cores": str(os.cpu_count() or "N/A"),
            "threads": str(os.cpu_count() or "N/A"),
        }]

    cpu_list = _wmi_query(wmi_conn, "Win32_Processor")

    for cpu in cpu_list:
        info = {
            "name": _safe_get(cpu, "Name", "Unknown").strip(),
            "manufacturer": _safe_get(cpu, "Manufacturer", "Unknown"),
            "description": _safe_get(cpu, "Description", "N/A"),
            "caption": _safe_get(cpu, "Caption", "N/A"),
            "architecture": _safe_get(cpu, "Architecture", None),
            "address_width": _safe_get(cpu, "AddressWidth", "N/A"),
            "data_width": _safe_get(cpu, "DataWidth", "N/A"),
            "max_clock_mhz": _safe_get(cpu, "MaxClockSpeed", None),
            "current_clock_mhz": _safe_get(cpu, "CurrentClockSpeed", None),
            "ext_clock_mhz": _safe_get(cpu, "ExtClock", None),
            "l2_cache_kb": _safe_get(cpu, "L2CacheSize", None),
            "l3_cache_kb": _safe_get(cpu, "L3CacheSize", None),
            "cores": _safe_get(cpu, "NumberOfCores", "N/A"),
            "threads": _safe_get(cpu, "NumberOfLogicalProcessors", "N/A"),
            "voltage": _safe_get(cpu, "CurrentVoltage", None),
            "socket": _safe_get(cpu, "SocketDesignation", "N/A"),
            "processor_id": _safe_get(cpu, "ProcessorId", "N/A"),
            "stepping": _safe_get(cpu, "Stepping", "N/A"),
            "revision": _safe_get(cpu, "Revision", "N/A"),
            "family": _safe_get(cpu, "Family", None),
            "processor_type": _safe_get(cpu, "ProcessorType", None),
            "load_percentage": _safe_get(cpu, "LoadPercentage", "N/A"),
            "virtualization_firmware": _safe_get(cpu, "VirtualizationFirmwareEnabled", None),
            "second_level_address_translation": _safe_get(cpu, "SecondLevelAddressTranslationExtensions", None),
            "vmm_monitor_mode_ext": _safe_get(cpu, "VMMonitorModeExtensions", None),
        }

        # Architecture map
        arch_map = {0: "x86", 1: "MIPS", 2: "Alpha", 3: "PowerPC", 5: "ARM",
                    6: "ia64", 9: "x64", 12: "ARM64"}
        arch_code = info["architecture"]
        if arch_code is not None:
            info["architecture_display"] = arch_map.get(int(arch_code), f"Unknown ({arch_code})")
        else:
            info["architecture_display"] = "N/A"

        # ProcessorType map
        ptype_map = {1: "Other", 2: "Unknown", 3: "Central Processor",
                     4: "Math Processor", 5: "DSP Processor", 6: "Video Processor"}
        ptype = info["processor_type"]
        if ptype is not None:
            info["package_display"] = ptype_map.get(int(ptype), f"Unknown ({ptype})")
        else:
            info["package_display"] = "N/A"

        # Code Name (heuristic from caption)
        caption = str(info.get("caption") or "").strip()
        name_upper = info["name"].upper()
        if "INTEL" in name_upper or "GENUINE INTEL" in name_upper:
            info["code_name"] = _guess_intel_codename(info["name"])
        elif "AMD" in name_upper:
            info["code_name"] = _guess_amd_codename(info["name"])
        else:
            info["code_name"] = caption or "N/A"

        # TDP (heuristic)
        info["tdp_display"] = _guess_cpu_tdp(info["name"])

        # Cache display
        l2 = info["l2_cache_kb"]
        l3 = info["l3_cache_kb"]
        cache_parts = []
        if l2:
            try:
                l2_int = int(l2)
                cache_parts.append(f"L2: {l2_int} KB ({l2_int/1024:.1f} MB)")
            except Exception:
                cache_parts.append(f"L2: {l2}")
        if l3:
            try:
                l3_int = int(l3)
                cache_parts.append(f"L3: {l3_int} KB ({l3_int/1024:.1f} MB)")
            except Exception:
                cache_parts.append(f"L3: {l3}")
        info["cache_display"] = " | ".join(cache_parts) if cache_parts else "N/A"

        # Max Clock display
        info["max_clock_display"] = _format_mhz(info["max_clock_mhz"])
        info["current_clock_display"] = _format_mhz(info["current_clock_mhz"])

        # Voltage / VID
        volt = info["voltage"]
        if volt is not None:
            try:
                volt_int = int(volt)
                if volt_int == 0:
                    info["vid_display"] = "N/A (Dynamic)"
                else:
                    info["vid_display"] = f"{volt_int} mV ({volt_int/1000:.3f} V)"
            except Exception:
                info["vid_display"] = str(volt)
        else:
            info["vid_display"] = "N/A"

        cpus.append(info)

    return cpus


def _guess_intel_codename(name: str) -> str:
    """Guess Intel CPU code name from model name."""
    n = name.upper()
    if "12TH" in n or "12TH GEN" in n or any(f"i{x}-12" in n for x in [3,5,7,9]):
        return "Alder Lake (12th Gen)"
    if "13TH" in n or "13TH GEN" in n or any(f"i{x}-13" in n for x in [3,5,7,9]):
        return "Raptor Lake (13th Gen)"
    if "14TH" in n or "14TH GEN" in n or any(f"i{x}-14" in n for x in [3,5,7,9]):
        return "Raptor Lake Refresh (14th Gen)"
    if "11TH" in n or any(f"i{x}-11" in n for x in [3,5,7,9]):
        return "Tiger Lake / Rocket Lake (11th Gen)"
    if "10TH" in n or any(f"i{x}-10" in n for x in [3,5,7,9]):
        return "Ice Lake / Comet Lake (10th Gen)"
    if "CORE ULTRA" in n:
        return "Meteor Lake / Arrow Lake (Core Ultra)"
    return "Intel (Code Name N/A)"


def _guess_amd_codename(name: str) -> str:
    """Guess AMD CPU code name from model name."""
    n = name.upper()
    if "RYZEN" in n:
        # Look for 4-digit number after Ryzen X
        import re
        m = re.search(r"RYZEN\s+\d\s+(\d{4})", n)
        if m:
            series = m.group(1)
            first_digit = series[0]
            if first_digit == "7":
                return "Ryzen 7000 Series (Zen 4)"
            if first_digit == "5":
                return "Ryzen 5000 Series (Zen 3)"
            if first_digit == "4":
                return "Ryzen 4000 Series (Zen 2)"
            if first_digit == "3":
                return "Ryzen 3000 Series (Zen 2 / Zen+)"
            if first_digit == "2":
                return "Ryzen 2000 Series (Zen+)"
            if first_digit == "1":
                return "Ryzen 1000 Series (Zen)"
        return "AMD Ryzen (Code Name N/A)"
    if "EPYC" in n:
        return "AMD EPYC"
    if "THREADRIPPER" in n:
        return "AMD Threadripper"
    return "AMD (Code Name N/A)"


def get_motherboard_info(wmi_conn) -> Dict[str, str]:
    """Get motherboard details."""
    info = {
        "manufacturer": "N/A",
        "model": "N/A",
        "version": "N/A",
        "serial": "N/A",
        "bus_specs": "N/A",
        "chipset": "N/A",
        "south_bridge": "N/A",
    }

    if not wmi_conn:
        return info

    # BaseBoard
    bb_list = _wmi_query(wmi_conn, "Win32_BaseBoard")
    if bb_list:
        bb = bb_list[0]
        info["manufacturer"] = _safe_get(bb, "Manufacturer", "N/A")
        info["model"] = _safe_get(bb, "Product", "N/A")
        info["version"] = _safe_get(bb, "Version", "N/A")
        info["serial"] = _safe_get(bb, "SerialNumber", "N/A")

    # Chipset — try to find from PnPEntity names
    try:
        pnp = wmi_conn.query(
            "SELECT Name FROM Win32_PnPEntity WHERE Name LIKE '%Chipset%' OR Name LIKE '%Host Bridge%'"
        )
        chipset_names = []
        for p in pnp:
            n = _safe_get(p, "Name", None)
            if n and "chipset" in n.lower():
                chipset_names.append(n)
        if chipset_names:
            info["chipset"] = chipset_names[0]
    except Exception:
        pass

    # Try to find chipset from registry
    if info["chipset"] == "N/A" and HAS_WINREG:
        try:
            info["chipset"] = _read_registry_chipset()
        except Exception:
            pass

    # Bus Specs — heuristic from CPU (PCIe generation)
    info["bus_specs"] = _guess_bus_specs(info.get("model", ""), "")

    return info


def _read_registry_chipset() -> str:
    """Read chipset from registry."""
    try:
        key_path = r"HARDWARE\DESCRIPTION\System\BIOS"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
        # Look for any chipset-related value
        i = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, i)
                if "chipset" in name.lower():
                    winreg.CloseKey(key)
                    return str(value)
                i += 1
            except OSError:
                break
        winreg.CloseKey(key)
    except Exception:
        pass
    return "N/A"


def _guess_bus_specs(mobo_model: str, chipset: str) -> str:
    """Guess bus specs from motherboard/chipset."""
    # PCIe generation is usually tied to CPU platform
    if "Z790" in chipset or "B760" in chipset or "H770" in chipset:
        return "PCIe 5.0 (CPU) + PCIe 4.0 (PCH)"
    if "Z690" in chipset or "B660" in chipset or "H670" in chipset:
        return "PCIe 5.0 (CPU) + PCIe 4.0 (PCH)"
    if "X670" in chipset or "B650" in chipset:
        return "PCIe 5.0 + PCIe 4.0"
    if "X570" in chipset or "B550" in chipset:
        return "PCIe 4.0"
    if "Z590" in chipset or "B560" in chipset:
        return "PCIe 4.0"
    return "PCIe (generation depends on CPU)"


def get_bios_info(wmi_conn) -> Dict[str, str]:
    """Get BIOS details."""
    info = {
        "brand": "N/A",
        "version": "N/A",
        "release_date": "N/A",
        "smbios_version": "N/A",
        "rom_size": "N/A",
        "serial": "N/A",
    }

    if not wmi_conn:
        return info

    bios_list = _wmi_query(wmi_conn, "Win32_BIOS")
    if bios_list:
        bios = bios_list[0]
        info["brand"] = _safe_get(bios, "Manufacturer", "N/A")
        info["version"] = _safe_get(bios, "SMBIOSBIOSVersion", "N/A")
        info["serial"] = _safe_get(bios, "SerialNumber", "N/A")

        # Release date
        date_raw = _safe_get(bios, "ReleaseDate", None)
        if date_raw:
            try:
                # Format: YYYYMMDD000000.000000+000
                date_str = str(date_raw)[:8]
                dt = datetime.strptime(date_str, "%Y%m%d")
                info["release_date"] = dt.strftime("%Y-%m-%d")
            except Exception:
                info["release_date"] = str(date_raw)

        # SMBIOS version (MajorVersion.MinorVersion)
        major = _safe_get(bios, "SMBIOSMajorVersion", None)
        minor = _safe_get(bios, "SMBIOSMinorVersion", None)
        if major is not None and minor is not None:
            info["smbios_version"] = f"{major}.{minor}"

        # ROM size (from BIOSCharacteristics or ExtendedBIOSDataTable)
        # WMI has no direct ROM size; we approximate from SMBIOS
        info["rom_size"] = _estimate_rom_size()

    return info


def _estimate_rom_size() -> str:
    """Estimate BIOS ROM size (typical values)."""
    # Typical modern UEFI: 16 MB or 32 MB
    return "16-32 MB (typical UEFI)"


def get_ram_info(wmi_conn) -> Dict[str, Any]:
    """Get RAM details."""
    info = {
        "total_gb": "N/A",
        "total_bytes": 0,
        "type": "N/A",
        "speed_mhz": "N/A",
        "manufacturer": "N/A",
        "slots": [],
        "num_slots_used": 0,
        "num_slots_total": 0,
        "form_factor": "N/A",
    }

    if HAS_PSUTIL:
        try:
            total = psutil.virtual_memory().total
            info["total_bytes"] = total
            info["total_gb"] = _format_bytes(total)
        except Exception:
            pass

    if not wmi_conn:
        return info

    # Physical Memory
    try:
        mem_list = _wmi_query(wmi_conn, "Win32_PhysicalMemory")
        slot_list = []
        types_set = set()
        speeds_set = set()
        manufacturers_set = set()
        form_factors_set = set()

        for mem in mem_list:
            capacity = _safe_get(mem, "Capacity", 0)
            try:
                capacity = int(capacity)
            except Exception:
                capacity = 0

            mem_type_code = _safe_get(mem, "SMBIOSMemoryType", None)
            mem_type = _map_memory_type(mem_type_code)

            speed = _safe_get(mem, "Speed", None)
            manufacturer = _safe_get(mem, "Manufacturer", "N/A")
            slot = _safe_get(mem, "DeviceLocator", "N/A")
            ff_code = _safe_get(mem, "FormFactor", None)
            ff = _map_form_factor(ff_code)

            slot_list.append({
                "slot": slot,
                "capacity": _format_bytes(capacity) if capacity else "N/A",
                "capacity_bytes": capacity,
                "type": mem_type,
                "speed_mhz": f"{speed} MHz" if speed else "N/A",
                "manufacturer": str(manufacturer).strip() if manufacturer else "N/A",
                "form_factor": ff,
            })

            if mem_type != "N/A":
                types_set.add(mem_type)
            if speed:
                speeds_set.add(int(speed))
            if manufacturer and manufacturer != "N/A":
                manufacturers_set.add(str(manufacturer).strip())
            if ff != "N/A":
                form_factors_set.add(ff)

        info["slots"] = slot_list
        info["num_slots_used"] = len(slot_list)

        if types_set:
            info["type"] = " / ".join(sorted(types_set))
        if speeds_set:
            info["speed_mhz"] = f"{max(speeds_set)} MHz"
        if manufacturers_set:
            info["manufacturer"] = " / ".join(sorted(manufacturers_set))
        if form_factors_set:
            info["form_factor"] = " / ".join(sorted(form_factors_set))

    except Exception as e:
        pass

    # Total slots
    try:
        arr = _wmi_query(wmi_conn, "Win32_PhysicalMemoryArray")
        if arr:
            total_slots = _safe_get(arr[0], "MemoryDevices", None)
            if total_slots:
                info["num_slots_total"] = int(total_slots)
    except Exception:
        pass

    return info


def _map_memory_type(code) -> str:
    """Map SMBIOSMemoryType code to name."""
    if code is None:
        return "N/A"
    try:
        code = int(code)
    except Exception:
        return "N/A"
    mapping = {
        0: "Unknown", 1: "Other", 2: "DRAM", 3: "Synchronous DRAM",
        4: "Cache DRAM", 5: "EDO", 6: "EDRAM", 7: "VRAM", 8: "SRAM",
        9: "RAM", 10: "ROM", 11: "Flash", 12: "EEPROM", 13: "FEPROM",
        14: "EPROM", 15: "CDRAM", 16: "3DRAM", 17: "SDRAM", 18: "SGRAM",
        19: "RDRAM", 20: "DDR", 21: "DDR2", 22: "DDR2 FB-DIMM",
        24: "DDR3", 25: "FBD2", 26: "DDR4", 27: "LPDDR",
        28: "LPDDR2", 29: "LPDDR3", 30: "LPDDR4",
        34: "DDR5", 35: "LPDDR5", 36: "HBM2", 37: "DDR6",
    }
    return mapping.get(code, f"Unknown ({code})")


def _map_form_factor(code) -> str:
    """Map FormFactor code to name."""
    if code is None:
        return "N/A"
    try:
        code = int(code)
    except Exception:
        return "N/A"
    mapping = {
        0: "Unknown", 1: "Other", 2: "SIP", 3: "DIP", 4: "ZIP",
        5: "SOJ", 6: "Proprietary", 7: "SIMM", 8: "DIMM",
        9: "TSOP", 10: "PGA", 11: "RIMM", 12: "SODIMM",
        13: "SRIMM", 14: "SMD", 15: "SSMP", 16: "QFP",
        17: "TQFP", 18: "SOIC", 19: "LCC", 20: "PLCC",
        21: "BGA", 22: "FPBGA", 23: "LGA",
    }
    return mapping.get(code, f"Unknown ({code})")


def get_gpu_info(wmi_conn) -> List[Dict[str, Any]]:
    """Get all GPUs."""
    gpus = []

    if not wmi_conn:
        return gpus

    try:
        vc_list = _wmi_query(wmi_conn, "Win32_VideoController")
        for vc in vc_list:
            name = _safe_get(vc, "Name", "Unknown")
            vram = _safe_get(vc, "AdapterRAM", None)
            try:
                vram_int = int(vram) if vram else None
            except Exception:
                vram_int = None

            driver_version = _safe_get(vc, "DriverVersion", "N/A")
            driver_date = _safe_get(vc, "DriverDate", None)
            video_processor = _safe_get(vc, "VideoProcessor", "N/A")
            adapter_compat = _safe_get(vc, "AdapterCompatibility", "N/A")
            current_res = _safe_get(vc, "VideoModeDescription", "N/A")
            refresh_rate = _safe_get(vc, "CurrentRefreshRate", None)
            status = _safe_get(vc, "Status", "N/A")
            availability = _safe_get(vc, "Availability", None)

            # Parse driver date
            driver_date_str = "N/A"
            if driver_date:
                try:
                    dt = datetime.strptime(str(driver_date)[:8], "%Y%m%d")
                    driver_date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    driver_date_str = str(driver_date)

            # GPU Type guess (integrated vs dedicated)
            gpu_type = _guess_gpu_type(name)

            info = {
                "name": name,
                "vram_bytes": vram_int,
                "vram_display": _format_bytes(vram_int) if vram_int else "N/A (or >4GB reported as 4GB)",
                "gpu_type": gpu_type,
                "manufacturer": adapter_compat,
                "video_processor": video_processor,
                "driver_version": driver_version,
                "driver_date": driver_date_str,
                "current_resolution": current_res,
                "refresh_rate": f"{refresh_rate} Hz" if refresh_rate else "N/A",
                "status": status,
            }
            gpus.append(info)
    except Exception as e:
        pass

    # Add NVIDIA detection via torch (if available)
    if HAS_TORCH and torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Check if already in list
                already = any(props.name in g["name"] for g in gpus)
                if not already:
                    gpus.append({
                        "name": props.name,
                        "vram_bytes": props.total_memory,
                        "vram_display": _format_bytes(props.total_memory),
                        "gpu_type": "Discrete (CUDA)",
                        "manufacturer": "NVIDIA",
                        "video_processor": props.name,
                        "driver_version": "N/A (via PyTorch)",
                        "driver_date": "N/A",
                        "current_resolution": "N/A",
                        "refresh_rate": "N/A",
                        "status": "OK (CUDA)",
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessors": props.multi_processor_count,
                    })
        except Exception:
            pass

    return gpus


def _guess_gpu_type(name: str) -> str:
    """Guess GPU type from name."""
    n = name.upper()
    # Integrated Intel
    if "INTEL" in n and ("UHD" in n or "HD GRAPHICS" in n or "IRIS" in n):
        return "Integrated (Intel)"
    if "INTEL" in n and "ARC" in n:
        return "Discrete (Intel Arc)"
    # AMD integrated
    if "AMD" in n and "RADEON" in n and ("VEGA" in n or "GRAPHICS" in n):
        return "Integrated (AMD APU)"
    # NVIDIA
    if "NVIDIA" in n or "GEFORCE" in n or "QUADRO" in n or "TESLA" in n:
        if any(x in n for x in ["MX", "GTX", "RTX", "QUADRO", "TESLA"]):
            return "Discrete (NVIDIA)"
        return "NVIDIA (unknown type)"
    # AMD dedicated
    if "AMD" in n and "RADEON" in n and any(x in n for x in ["RX", "PRO", "VEGA"]):
        return "Discrete (AMD Radeon)"
    # Virtual
    if any(x in n for x in ["VIRTUAL", "VMWARE", "VIRTUALBOX", "MICROSOFT BASIC"]):
        return "Virtual / Basic"
    return "Unknown"


# ================================================================
# ✅ Rich Display Functions
# ================================================================

def _display_rich(system, cpus, mobo, bios, ram, gpus):
    """Display all info using Rich."""

    console.print()
    console.print(Panel.fit(
        "💻  HARDWARE DETECTION REPORT  💻\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="bright_cyan",
        padding=(1, 4),
    ))

    # ============================================================
    # SYSTEM PANEL
    # ============================================================
    system_table = Table(box=box.ROUNDED, show_header=False, border_style="bright_blue")
    system_table.add_column("Field", style="bold cyan", width=22)
    system_table.add_column("Value", style="white")

    system_table.add_row("🏢 Brand", f"[bold yellow]{system['manufacturer_display']}[/bold yellow]")
    system_table.add_row("📦 Model", f"[bold white]{system['model']}[/bold white]")
    system_table.add_row("💻 Chassis", system["chassis"])
    system_table.add_row("🖥️  System Type", system["system_type"])
    system_table.add_row("🔢 Serial", f"[dim]{system['serial']}[/dim]")
    system_table.add_row("🏷️  Hostname", system["hostname"])
    system_table.add_row("🪟 OS", system["os_name"])
    system_table.add_row("🐍 Python", system["python_version"])

    console.print(Panel(system_table, title="[bold magenta]System Information[/bold magenta]",
                        border_style="magenta"))

    # ============================================================
    # CPU PANEL
    # ============================================================
    for i, cpu in enumerate(cpus, 1):
        title = f"[bold magenta]CPU #{i}[/bold magenta]" if len(cpus) > 1 else "[bold magenta]CPU Information[/bold magenta]"
        cpu_table = Table(box=box.ROUNDED, show_header=False, border_style="bright_green")
        cpu_table.add_column("Field", style="bold cyan", width=26)
        cpu_table.add_column("Value", style="white")

        cpu_table.add_row("🔤 Processor Name", f"[bold yellow]{cpu['name']}[/bold yellow]")
        cpu_table.add_row("🏷️  Code Name", cpu["code_name"])
        cpu_table.add_row("⚡ Max TDP", f"[bright_red]{cpu['tdp_display']}[/bright_red]")
        cpu_table.add_row("📦 Package", cpu["package_display"])
        cpu_table.add_row("⚙️  Technology", f"{cpu['architecture_display']} / {cpu['data_width']}-bit")
        cpu_table.add_row("🔌 Core VID", cpu["vid_display"])
        cpu_table.add_row("📋 Specification", cpu["description"])
        cpu_table.add_row("💾 Cache", cpu["cache_display"])
        cpu_table.add_row("🔢 Cores", f"[bold green]{cpu['cores']}[/bold green]")
        cpu_table.add_row("🧵 Threads", f"[bold green]{cpu['threads']}[/bold green]")
        cpu_table.add_row("⏱️  Max Clock", cpu["max_clock_display"])
        cpu_table.add_row("⏱️  Current Clock", cpu["current_clock_display"])
        cpu_table.add_row("🔲 Socket", cpu["socket"])
        cpu_table.add_row("🎯 Stepping", str(cpu["stepping"]))
        cpu_table.add_row("📊 Load", f"{cpu['load_percentage']}%")

        console.print(Panel(cpu_table, title=title, border_style="green"))

    # ============================================================
    # MOTHERBOARD PANEL
    # ============================================================
    mb_table = Table(box=box.ROUNDED, show_header=False, border_style="bright_yellow")
    mb_table.add_column("Field", style="bold cyan", width=26)
    mb_table.add_column("Value", style="white")

    mb_table.add_row("🏭 Manufacturer", f"[bold yellow]{mobo['manufacturer']}[/bold yellow]")
    mb_table.add_row("📦 Model", mobo["model"])
    mb_table.add_row("🔖 Version", mobo["version"])
    mb_table.add_row("🔢 Serial", f"[dim]{mobo['serial']}[/dim]")
    mb_table.add_row("🚌 Bus Specs", mobo["bus_specs"])
    mb_table.add_row("🧩 Chipset", mobo["chipset"])
    mb_table.add_row("🌉 South Bridge", mobo["south_bridge"])

    console.print(Panel(mb_table, title="[bold yellow]Motherboard Information[/bold yellow]",
                        border_style="yellow"))

    # ============================================================
    # BIOS PANEL
    # ============================================================
    bios_table = Table(box=box.ROUNDED, show_header=False, border_style="bright_red")
    bios_table.add_column("Field", style="bold cyan", width=26)
    bios_table.add_column("Value", style="white")

    bios_table.add_row("🏭 BIOS Brand", f"[bold yellow]{bios['brand']}[/bold yellow]")
    bios_table.add_row("📌 BIOS Version", bios["version"])
    bios_table.add_row("📅 Release Date", bios["release_date"])
    bios_table.add_row("🔧 SMBIOS Version", bios["smbios_version"])
    bios_table.add_row("💾 ROM Size", bios["rom_size"])

    console.print(Panel(bios_table, title="[bold red]BIOS Information[/bold red]",
                        border_style="red"))

    # ============================================================
    # MEMORY PANEL
    # ============================================================
    mem_table = Table(box=box.ROUNDED, show_header=False, border_style="bright_blue")
    mem_table.add_column("Field", style="bold cyan", width=26)
    mem_table.add_column("Value", style="white")

    mem_table.add_row("🧠 Total RAM", f"[bold green]{ram['total_gb']}[/bold green]")
    mem_table.add_row("🏷️  RAM Type", ram["type"])
    mem_table.add_row("⚡ Speed", ram["speed_mhz"])
    mem_table.add_row("🏭 Manufacturer", ram["manufacturer"])
    mem_table.add_row("📏 Form Factor", ram["form_factor"])
    mem_table.add_row("🔢 Slots Used", f"{ram['num_slots_used']} / {ram['num_slots_total'] or 'N/A'}")
    mem_table.add_row("💾 ROM", "See BIOS section")

    console.print(Panel(mem_table, title="[bold blue]Memory Information[/bold blue]",
                        border_style="blue"))

    # RAM slots table
    if ram["slots"]:
        slots_table = Table(title="[bold cyan]RAM Slots Details[/bold cyan]",
                             box=box.ROUNDED, border_style="cyan")
        slots_table.add_column("Slot", style="bold yellow")
        slots_table.add_column("Capacity", style="green")
        slots_table.add_column("Type", style="cyan")
        slots_table.add_column("Speed", style="magenta")
        slots_table.add_column("Manufacturer", style="white")
        slots_table.add_column("Form Factor", style="blue")

        for slot in ram["slots"]:
            slots_table.add_row(
                str(slot["slot"]),
                str(slot["capacity"]),
                str(slot["type"]),
                str(slot["speed_mhz"]),
                str(slot["manufacturer"])[:30],
                str(slot["form_factor"]),
            )

        console.print(slots_table)

    # ============================================================
    # GPU PANEL
    # ============================================================
    if not gpus:
        console.print(Panel("[red]No GPU detected[/red]",
                             title="[bold red]GPU Information[/bold red]",
                             border_style="red"))
    else:
        for i, gpu in enumerate(gpus, 1):
            title = f"[bold magenta]GPU #{i}: {gpu['name']}[/bold magenta]"
            gpu_table = Table(box=box.ROUNDED, show_header=False, border_style="bright_magenta")
            gpu_table.add_column("Field", style="bold cyan", width=26)
            gpu_table.add_column("Value", style="white")

            gpu_table.add_row("🎮 GPU Name", f"[bold yellow]{gpu['name']}[/bold yellow]")
            gpu_table.add_row("💾 VRAM Size", f"[bold green]{gpu['vram_display']}[/bold green]")
            gpu_table.add_row("🏷️  GPU Type", f"[bold cyan]{gpu['gpu_type']}[/bold cyan]")
            gpu_table.add_row("🏭 Manufacturer", gpu["manufacturer"])
            gpu_table.add_row("🔧 Video Processor", gpu.get("video_processor", "N/A"))
            gpu_table.add_row("📌 Driver Version", gpu.get("driver_version", "N/A"))
            gpu_table.add_row("📅 Driver Date", gpu.get("driver_date", "N/A"))
            gpu_table.add_row("🖥️  Current Resolution", gpu.get("current_resolution", "N/A"))
            gpu_table.add_row("🔄 Refresh Rate", gpu.get("refresh_rate", "N/A"))
            gpu_table.add_row("✅ Status", gpu.get("status", "N/A"))

            if "compute_capability" in gpu:
                gpu_table.add_row("🔢 Compute Capability", gpu["compute_capability"])
                gpu_table.add_row("🧮 Multiprocessors", str(gpu["multiprocessors"]))

            console.print(Panel(gpu_table, title=title, border_style="magenta"))

    # ============================================================
    # SUMMARY PANEL
    # ============================================================
    console.print()
    summary_text = Text()
    summary_text.append("✅ ", style="bold green")
    summary_text.append("Hardware detection complete.\n", style="bold white")
    summary_text.append(f"   Brand: ", style="cyan")
    summary_text.append(f"{system['manufacturer_display']} {system['model']}\n", style="bold yellow")
    if cpus:
        summary_text.append(f"   CPU:   ", style="cyan")
        summary_text.append(f"{cpus[0]['name']}\n", style="bold yellow")
    summary_text.append(f"   RAM:   ", style="cyan")
    summary_text.append(f"{ram['total_gb']}\n", style="bold yellow")
    if gpus:
        summary_text.append(f"   GPU:   ", style="cyan")
        summary_text.append(f"{', '.join(g['name'] for g in gpus)}\n", style="bold yellow")

    console.print(Panel(summary_text, title="[bold green]Summary[/bold green]",
                        border_style="green"))


# ================================================================
# ✅ Fallback (ANSI) Display
# ================================================================

def _print_header_ansi():
    """Print header with ANSI."""
    C = Colors
    print()
    print(f"{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗{C.ENDC}")
    print(f"{C.CYAN}{C.BOLD}║      💻  HARDWARE DETECTION REPORT  💻                   ║{C.ENDC}")
    print(f"{C.CYAN}{C.BOLD}║      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                ║{C.ENDC}")
    print(f"{C.CYAN}{C.BOLD}╚══════════════════════════════════════════════════════════╝{C.ENDC}")


def _print_section_ansi(title: str, color: str = Colors.CYAN):
    C = Colors
    print()
    print(f"{color}{C.BOLD}{'='*70}{C.ENDC}")
    print(f"{color}{C.BOLD}  {title}{C.ENDC}")
    print(f"{color}{C.BOLD}{'='*70}{C.ENDC}")


def _print_field_ansi(label: str, value: str, label_color: str = Colors.CYAN):
    C = Colors
    print(f"  {label_color}{label:<30}{C.ENDC} : {C.YELLOW}{value}{C.ENDC}")


def _display_ansi(system, cpus, mobo, bios, ram, gpus):
    """Display all info using ANSI (no Rich)."""
    _print_header_ansi()

    # System
    _print_section_ansi("🖥️  SYSTEM INFORMATION", Colors.MAGENTA)
    _print_field_ansi("Brand", system["manufacturer_display"], Colors.MAGENTA)
    _print_field_ansi("Model", system["model"], Colors.MAGENTA)
    _print_field_ansi("Chassis", system["chassis"], Colors.MAGENTA)
    _print_field_ansi("System Type", system["system_type"], Colors.MAGENTA)
    _print_field_ansi("Serial", system["serial"], Colors.MAGENTA)
    _print_field_ansi("Hostname", system["hostname"], Colors.MAGENTA)
    _print_field_ansi("OS", system["os_name"], Colors.MAGENTA)
    _print_field_ansi("Python", system["python_version"], Colors.MAGENTA)

    # CPU
    for i, cpu in enumerate(cpus, 1):
        _print_section_ansi(f"🧠 CPU #{i}", Colors.GREEN)
        _print_field_ansi("Processor Name", cpu["name"], Colors.GREEN)
        _print_field_ansi("Code Name", cpu["code_name"], Colors.GREEN)
        _print_field_ansi("Max TDP", cpu["tdp_display"], Colors.RED)
        _print_field_ansi("Package", cpu["package_display"], Colors.GREEN)
        _print_field_ansi("Technology", f"{cpu['architecture_display']} / {cpu['data_width']}-bit", Colors.GREEN)
        _print_field_ansi("Core VID", cpu["vid_display"], Colors.GREEN)
        _print_field_ansi("Specification", cpu["description"], Colors.GREEN)
        _print_field_ansi("Cache", cpu["cache_display"], Colors.GREEN)
        _print_field_ansi("Number of Cores", str(cpu["cores"]), Colors.GREEN)
        _print_field_ansi("Number of Threads", str(cpu["threads"]), Colors.GREEN)
        _print_field_ansi("Max Clock", cpu["max_clock_display"], Colors.GREEN)
        _print_field_ansi("Current Clock", cpu["current_clock_display"], Colors.GREEN)
        _print_field_ansi("Socket", cpu["socket"], Colors.GREEN)

    # Motherboard
    _print_section_ansi("🔌 MOTHERBOARD", Colors.YELLOW)
    _print_field_ansi("Manufacturer", mobo["manufacturer"], Colors.YELLOW)
    _print_field_ansi("Model", mobo["model"], Colors.YELLOW)
    _print_field_ansi("Version", mobo["version"], Colors.YELLOW)
    _print_field_ansi("Bus Specs", mobo["bus_specs"], Colors.YELLOW)
    _print_field_ansi("Chipset", mobo["chipset"], Colors.YELLOW)
    _print_field_ansi("South Bridge", mobo["south_bridge"], Colors.YELLOW)

    # BIOS
    _print_section_ansi("💾 BIOS", Colors.RED)
    _print_field_ansi("BIOS Brand", bios["brand"], Colors.RED)
    _print_field_ansi("BIOS Version", bios["version"], Colors.RED)
    _print_field_ansi("Release Date", bios["release_date"], Colors.RED)
    _print_field_ansi("SMBIOS Version", bios["smbios_version"], Colors.RED)
    _print_field_ansi("ROM Size", bios["rom_size"], Colors.RED)

    # Memory
    _print_section_ansi("🧠 MEMORY", Colors.BLUE)
    _print_field_ansi("Total RAM", ram["total_gb"], Colors.BLUE)
    _print_field_ansi("RAM Type", ram["type"], Colors.BLUE)
    _print_field_ansi("Speed", ram["speed_mhz"], Colors.BLUE)
    _print_field_ansi("Manufacturer", ram["manufacturer"], Colors.BLUE)
    _print_field_ansi("Form Factor", ram["form_factor"], Colors.BLUE)
    _print_field_ansi("Slots Used", f"{ram['num_slots_used']} / {ram['num_slots_total'] or 'N/A'}", Colors.BLUE)

    # RAM Slots
    if ram["slots"]:
        print()
        print(f"  {Colors.CYAN}{Colors.BOLD}RAM Slots Details:{Colors.ENDC}")
        for i, slot in enumerate(ram["slots"], 1):
            print(f"    [{i}] {Colors.YELLOW}{slot['slot']}{Colors.ENDC}: "
                  f"{slot['capacity']} | {slot['type']} | {slot['speed_mhz']} | {slot['manufacturer']}")

    # GPU
    for i, gpu in enumerate(gpus, 1):
        _print_section_ansi(f"🎮 GPU #{i}", Colors.MAGENTA)
        _print_field_ansi("GPU Name", gpu["name"], Colors.MAGENTA)
        _print_field_ansi("VRAM Size", gpu["vram_display"], Colors.MAGENTA)
        _print_field_ansi("GPU Type", gpu["gpu_type"], Colors.MAGENTA)
        _print_field_ansi("Manufacturer", gpu["manufacturer"], Colors.MAGENTA)
        _print_field_ansi("Video Processor", gpu.get("video_processor", "N/A"), Colors.MAGENTA)
        _print_field_ansi("Driver Version", gpu.get("driver_version", "N/A"), Colors.MAGENTA)
        _print_field_ansi("Driver Date", gpu.get("driver_date", "N/A"), Colors.MAGENTA)
        _print_field_ansi("Resolution", gpu.get("current_resolution", "N/A"), Colors.MAGENTA)
        _print_field_ansi("Refresh Rate", gpu.get("refresh_rate", "N/A"), Colors.MAGENTA)
        _print_field_ansi("Status", gpu.get("status", "N/A"), Colors.MAGENTA)
        if "compute_capability" in gpu:
            _print_field_ansi("Compute Capability", gpu["compute_capability"], Colors.MAGENTA)
            _print_field_ansi("Multiprocessors", str(gpu["multiprocessors"]), Colors.MAGENTA)

    print()
    print(f"{Colors.GREEN}{Colors.BOLD}✅ Hardware detection complete.{Colors.ENDC}")


# ================================================================
# ✅ Main Entry (called from SSLAlgorithms.py option 16)
# ================================================================

def _hw_():
    """
    Main entry point for hardware detection (option 16).

    Usage:
        from src.findDevice.hardwareDetector import _hw_
        _hw_()
    """
    if HAS_RICH:
        console.print(Panel.fit(
            "💻  Detecting Hardware...  💻",
            border_style="bright_cyan",
            padding=(1, 4),
        ))

    # ============================================================
    # Initialize WMI
    # ============================================================
    wmi_conn = None
    if HAS_WMI:
        try:
            wmi_conn = wmi.WMI()
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]⚠️ Could not initialize WMI: {e}[/yellow]")
            else:
                print(f"{Colors.YELLOW}⚠️ Could not initialize WMI: {e}{Colors.ENDC}")

    # ============================================================
    # Collect all info
    # ============================================================
    try:
        system = get_system_info(wmi_conn)
        cpus = get_cpu_info(wmi_conn)
        mobo = get_motherboard_info(wmi_conn)
        bios = get_bios_info(wmi_conn)
        ram = get_ram_info(wmi_conn)
        gpus = get_gpu_info(wmi_conn)
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error collecting hardware info: {e}[/red]")
        else:
            print(f"{Colors.RED}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================
    # Display
    # ============================================================
    if HAS_RICH:
        _display_rich(system, cpus, mobo, bios, ram, gpus)
    else:
        _display_ansi(system, cpus, mobo, bios, ram, gpus)


# Backward-compat alias
_hw_detect = _hw_


if __name__ == "__main__":
    _hw_()
