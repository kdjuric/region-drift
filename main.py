from region_drift import *

if __name__ == "__main__":
    cal_exists, list_of_files = check_cal_folder()
    no_of_files = len(list_of_files)

    for file_index in range(no_of_files):
        flux_finder = FindFluxes(list_of_files[file_index] + ".fit",
            list_of_files[file_index] + ".reg", cal_exists)
        
        flux_finder.main()
        print(f"Done file {file_index + 1} of {no_of_files}")

        flux_finder.find_exposure_combos()