"""
# For each file
    # Load data
        # if .tif or .am require .am coord or _mask.tif _mask.am image exist
        # if .tif, .mrc, .rec require .csv coord or _mask.tif, _mask.mrc, _mask.rec exist
    # If coordinate file exist load coordinate file
        # Check if coordinates are compatible with image shape
        # Build semantic mask from coordinates
    # Voxalize image to fit needed image size
        # Save in train/imgs
    # Voxalize mask to fir needed mask size
        # Save in train/masks
"""
