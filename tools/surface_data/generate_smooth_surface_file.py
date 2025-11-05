import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

def interpolate_and_save_surface_data(input_filename, output_filename, fine_grid_resolution=200):
    """
    Load surface data from an HDF5 file, perform spline interpolation over a finer grid,
    and save the interpolated data to a new HDF5 file.

    Parameters:
    - input_filename (str): Path to the input HDF5 file containing original surface data.
    - output_filename (str): Path to save the output HDF5 file with interpolated data.
    - fine_grid_resolution (int): Number of points for the fine grid interpolation.

    """
    # Load data
    with h5py.File(input_filename, "r") as data:
        print("Available datasets:", list(data.keys()))
        energy = data['E'][:]
        angle = data['A'][:]
        spyld = data['spyld'][:]
        rfyld = data['rfyld'][:]

    print("Energy shape:", energy.shape)
    print("Angle shape:", angle.shape)
    print("Sputtering Yield (spyld) shape:", spyld.shape)
    print("Reflection Yield (rfyld) shape:", rfyld.shape)

    # Perform spline interpolation for smoother results
    spyld_interp = RectBivariateSpline(energy, angle, spyld)
    rfyld_interp = RectBivariateSpline(energy, angle, rfyld)

    # Define a finer grid for smooth interpolation
    energy_fine = np.linspace(energy.min(), energy.max(), fine_grid_resolution)
    angle_fine = np.linspace(angle.min(), angle.max(), fine_grid_resolution)

    # Interpolated data on the finer grid
    spyld_smooth = spyld_interp(energy_fine, angle_fine)
    rfyld_smooth = rfyld_interp(energy_fine, angle_fine)

    # Save interpolated data to a new HDF5 file
    with h5py.File(output_filename, "w") as new_file:
        new_file.create_dataset('E', data=energy_fine)
        new_file.create_dataset('A', data=angle_fine)
        new_file.create_dataset('spyld', data=spyld_smooth)
        new_file.create_dataset('rfyld', data=rfyld_smooth)

    print(f"Interpolated data saved to {output_filename}")

if __name__ == "__main__":
    input_filename = "74_on_74.h5"
    output_filename = "74_on_74_interpolated_fine_grid.h5"
    interpolate_and_save_surface_data(input_filename, output_filename)


