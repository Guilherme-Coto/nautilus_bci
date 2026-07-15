import os

header_paths = [
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI.h",
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI_gHIamp.h",
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI_gNautilus.h",
    r"C:\Users\VR\Documents\gtec\gNEEDaccessClientAPI\C\GDSClientAPI_gUSBamp.h"
]

for path in header_paths:
    if os.path.exists(path):
        print(f"=== File: {path} ===")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        # Search for lines containing impedance, error, or negative definitions
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # look for constants or comments with negative values
            if 'impedance' in line_lower or 'error' in line_lower or 'status' in line_lower:
                if any(x in line for x in ['-', '0x', 'define', 'const', 'enum']):
                    print(f"Line {i+1}: {line.strip()}")
    else:
        print(f"Not found: {path}")
