"""
Combine 3DGRUT neural volume (USDZ) and nvblox mesh (USD) into a single aligned stage.

This script:
1. Reads the transform from the USDZ's gauss.usda
2. Applies the same transform to the mesh so they align
3. Creates a combined stage with both assets

Usage in Isaac Sim Script Editor:
    exec(open("G:/GithubProject/ReconS/utilities/issacsim_visualize_sancheck.py").read())
"""

import os
import zipfile
import tempfile
import omni.usd
from pxr import Usd, UsdGeom, Sdf, Gf

# =============================================================================
# Configuration - Update these paths for your data
# =============================================================================
BASE_DIR = r"G:/GithubProject/ReconS/data/sample_20260119_i4"
MESH_USD_PATH = os.path.join(BASE_DIR, "nvblox_sfm_out", "nvblox_mesh.usd")
VOLUME_USDZ_PATH = os.path.join(
    BASE_DIR,
    "3dgrut_out",
    "sample_20260119_i4_3dgut_norm",
    "3dgrut_data-0702_231117",
    "export_last.usdz"
)
OUTPUT_STAGE_PATH = os.path.join(BASE_DIR, "combined_stage.usda")


def extract_transform_from_usdz(usdz_path):
    """
    Extract the transform matrix from the gauss.usda inside the USDZ file.

    Args:
        usdz_path: Path to the USDZ file

    Returns:
        Gf.Matrix4d: The transform matrix, or identity if not found
    """
    print(f"Reading transform from: {usdz_path}")

    try:
        with zipfile.ZipFile(usdz_path, 'r') as zf:
            # Find the gauss.usda file
            gauss_files = [f for f in zf.namelist() if 'gauss' in f.lower() and f.endswith('.usda')]

            if not gauss_files:
                print("Warning: No gauss.usda found in USDZ, using identity transform")
                return Gf.Matrix4d(1.0)

            gauss_file = gauss_files[0]
            print(f"Found: {gauss_file}")

            # Extract to temp file and open as USD stage
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extract(gauss_file, tmpdir)
                gauss_path = os.path.join(tmpdir, gauss_file)

                # Open the extracted USD file
                gauss_stage = Usd.Stage.Open(gauss_path)
                if not gauss_stage:
                    print("Warning: Could not open gauss.usda, using identity transform")
                    return Gf.Matrix4d(1.0)

                # Find the Volume or Xform prim with transform
                for prim in gauss_stage.Traverse():
                    if prim.IsA(UsdGeom.Xformable):
                        xformable = UsdGeom.Xformable(prim)
                        xform_ops = xformable.GetOrderedXformOps()

                        for op in xform_ops:
                            if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                                transform = op.Get()
                                print(f"Found transform at {prim.GetPath()}")
                                print(f"Transform matrix:\n{transform}")
                                return transform

                print("Warning: No transform found in gauss.usda, using identity")
                return Gf.Matrix4d(1.0)

    except Exception as e:
        print(f"Error reading USDZ: {e}")
        print("Using identity transform")
        return Gf.Matrix4d(1.0)


def create_combined_stage(mesh_path, volume_path, output_path, apply_transform=True):
    """
    Create a combined USD stage with both mesh and volume aligned.

    Args:
        mesh_path: Path to the nvblox mesh USD file
        volume_path: Path to the 3dgrut volume USDZ file
        output_path: Path to save the combined stage
        apply_transform: If True, apply volume's transform to mesh for alignment
    """
    # Get current stage
    stage = omni.usd.get_context().get_stage()

    # Clear existing prims if re-running
    if stage.GetPrimAtPath("/World"):
        stage.RemovePrim("/World")

    # Define world root
    world = stage.DefinePrim("/World", "Xform")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetDefaultPrim(world)

    # Add nvblox mesh as reference
    print(f"Adding mesh: {mesh_path}")
    mesh_prim = stage.DefinePrim("/World/mesh", "Xform")
    mesh_prim.GetReferences().AddReference(mesh_path)

    # Add 3dgrut volume as reference
    print(f"Adding volume: {volume_path}")
    volume_prim = stage.DefinePrim("/World/volume", "Xform")
    volume_prim.GetReferences().AddReference(volume_path)

    # Apply transform to mesh to align with volume
    if apply_transform:
        transform = extract_transform_from_usdz(volume_path)

        xformable = UsdGeom.Xformable(mesh_prim)
        xformable.ClearXformOpOrder()
        transform_op = xformable.AddTransformOp()
        transform_op.Set(transform)
        print("Applied volume transform to mesh for alignment")

    # Save the combined stage
    stage.GetRootLayer().Export(output_path)
    print(f"\nSaved combined stage to: {output_path}")
    print("Both mesh and volume should now be aligned!")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__" or True:  # Always run when executed in Isaac Sim
    print("=" * 60)
    print("Combining 3DGRUT Volume + nvblox Mesh (Aligned)")
    print("=" * 60)

    # Validate paths
    if not os.path.exists(MESH_USD_PATH):
        print(f"Error: Mesh not found: {MESH_USD_PATH}")
    elif not os.path.exists(VOLUME_USDZ_PATH):
        print(f"Error: Volume not found: {VOLUME_USDZ_PATH}")
    else:
        create_combined_stage(
            mesh_path=MESH_USD_PATH,
            volume_path=VOLUME_USDZ_PATH,
            output_path=OUTPUT_STAGE_PATH,
            apply_transform=True
        )