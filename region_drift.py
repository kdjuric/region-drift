import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from regions.core import PixCoord
from regions.shapes.circle import CirclePixelRegion

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

class FindFluxes:
    exposure_combos = []

    def __init__(self, fit_file, reg, calibration_folder=False) -> None:
        self.output_data = []
        self.list_of_points = []
        self.list_of_calibrated_points = []

        self.data = self.mask = self.aperture = self.hdu = []
        self.total_sum = self.weighted_data = []

        self.cal_folder = calibration_folder

        self.fits = fit_file
        self.reg = reg
        self.cal_reg = (CAL_FOLDER + reg.split('/')[-1])

        hdu_list = fits.open(self.fits)
        self.date_time = hdu_list[0].header['DATE-OBS'].replace('T', '')
        self.exposure = int(np.floor(float(hdu_list[0].header['EXPTIME'])))
        self.filter = hdu_list[0].header['FILTER']

    def open_reg(self, reg_file, calibrated) -> None:
        with open(reg_file, 'r') as temp_reg_file:
            for line in temp_reg_file:
                if line.startswith("circle"):
                    _temp_list = line[7:].split(')')
                    values = _temp_list[0].split(',')
            
        if calibrated:
            self.list_of_calibrated_points.append(values[0], 
                values[1], values[2], _temp_list[1])
        
        else:
            self.list_of_points.append(
                (np.round(float(values[0]), 6),
                np.round(float(values[1]), 6),
                np.round(float(values[2]), 6),
                _temp_list[1])
            )

    def output_to_file(self, name, list_of_points, 
        list_of_calibrated_points, list_of_sums) -> None:
        
        with open(name + f"_DATA_{self.filter}{self.exposure}.txt", 'w') as temp_text_file:
            if self.exposure not in self.exposure_combos:
                self.exposure_combos.append(self.exposure)

            temp_text_file.write(f"Header: date-time: \t {self.date_time} \t " \
                f"Exposure: \t {self.exposure} \t Filter: \t {self.filter}\n")

            temp_text_file.write("Header: \t x-axis \t y-axis \t radius \t sum " \
                "\t error \t mag (-2.5log(sum)) \t az \t alt \t zen \n")

            for i, sum, cal_i in zip(list_of_points, list_of_sums, list_of_calibrated_points):
                az, alt = self.ra_dec_to_alt_az(cal_i[0], cal_i[1])

                err_in_sum = np.round(np.sqrt(sum), 6)
                temp_text_file.write(f"{i[0]} \t {i[1]} \t {i[2]} \t "
                f"{sum} \t {err_in_sum} \t {np.round(-2.5*np.log10(sum), 6)} \t "
                f"{az} \t {alt} \t {np.round(90-float(alt), 6)}\n")

    def ra_dec_to_alt_az(self, region_ra, region_dec) -> str:
        
        c = SkyCoord(ra=region_ra, dec=region_dec, unit=(u.hourangle, u.deg), frame='fk5')
        loc = EarthLocation(lat=43.4643, lon=-80.5204, height=340*u.m)
        time = Time(self.date_time)
        cAltAz = c.transform_to(AltAz(obstime=time, location=loc))

        return cAltAz.to_string(style='decimal').split(" ")

    def find_sum(self, entry) -> None:

        x_cord, y_cord, radius, _other = entry
        self.aperture = CirclePixelRegion(PixCoord(x_cord, y_cord), radius)

        self.mask = self.aperture.to_mask(mode='exact')
        self.hdu = fits.open(self.fits)[0]
        self.data = self.mask.cutout(self.hdu.data)

        self.weighted_data = self.mask.multiply(self.hdu.data)
        self.total_sum = np.round(np.sum(self.weighted_data), 1)
        self.output_data.append(self.total_sum)

    def main(self) -> None:
        self.open_reg(self.reg, False)

        if self.cal_folder:
            self.open_reg(self.cal_reg, True)

        for point in self.list_of_points:
            self.find_sum(point)
            self.output_to_file(self.fits[:-4],
                self.list_of_points,
                self.list_of_calibrated_points,
                self.output_data
            )

    def find_exposure_combos(self) -> None:
        text_files = []

        for i in self.exposure_combos:
            pair = glob.glob(FOLDER + f"*{i}.txt")

            if not text_files:
                text_files.append(pair)
            else:
                for i in text_files:
                    if not set(pair) == set(i):
                        text_files.append(pair)
                        break
        
        self.colour_mags_to_text(text_files)

    @staticmethod
    def find_calibrations(b, v, zen_b, zen_v) -> tuple:
        V = v - V_KAPPA * (1 / np.cos(zen_v)) + V_ALPHA * (b - v) + V_BETA
        B = b - B_KAPPA * (1 / np.cos(zen_b)) + B_ALPHA * (b - v) + B_BETA

        return B, V

    def b_v_finder(self, file1, file2):
        self.colour_mags = []

        with open(file1) as b_file, open(file2) as v_file:
            for b_line, v_line in zip(b_file, v_file):
                if "Exposure:" in b_line:
                    exposure = str(b_line.split("\t")[3]).replace(" ", "")
                if "Header:" not in b_line:
                    b_line = b_line.split("\t")
                    v_line = v_line.split("\t")
                    b_sum = float(b_line[3])
                    v_sum = float(v_line[3])
                    b_sum_error = float(b_line[4])
                    v_sum_error = float(v_line[4])
                    b = float(b_line[5])
                    v = float(v_line[5])
                    zen_b = float(b_line[8])
                    zen_v = float(v_line[8])

                v_error = self.find_v_error(v_sum, b_sum, v_sum_error, b_sum_error, b, v, zen_b, zen_v)
                b_error = self.find_b_error(v_sum, b_sum, v_sum_error, b_sum_error, b, v, zen_b, zen_v)

                b_minus_v_error = (b - v) * np.sqrt((v_error/v)**2  + (b_error/b)**2)
                B, V = self.find_calibrations(b, v, zen_b, zen_v)
                self.colour_mags.append((B, V, B-V, v_error, b_minus_v_error))

        return exposure

    @staticmethod
    def find_b_error(v_sum, b_sum, v_sum_error, b_sum_error, b, v, zen_b, zen_v) -> float:
        first_term = (-2.5 * (b_sum_error / (np.log(10) * b_sum)))**2
        second_term = (3.3 * (1/np.cos(zen_b)) * np.sqrt((0.45/3.3)**2 + (((1/np.cos(zen_b)) * np.tan(zen_b) * 0.01)/ (1/np.cos(zen_b)))**2))**2

        third_term_a = (0.223 * (-2.5 * np.log10(b_sum/v_sum)))
        fun1 = (v_sum / (b_sum * np.log(10))) * np.sqrt((b_sum_error/b_sum)**2 + (v_sum_error/v_sum)**2)

        third_term_b = np.sqrt((0.06/0.223)**2 + (fun1/np.log10(b_sum/v_sum))**2)
        third_term_all = (third_term_a * third_term_b)**2
        
        final_error = np.sqrt(first_term + second_term + third_term_all + 0.980**2)

        return final_error

    @staticmethod
    def find_v_error(v_sum, b_sum, v_sum_error, b_sum_error, b, v, zen_b, zen_v) -> float:
        first_term = (-2.5 * (v_sum_error / (np.log(10) * v_sum)))**2
        second_term = (2.706 * (1/np.cos(zen_v)) * np.sqrt((0.39/2.706)**2 + (((1/np.cos(zen_v)) * np.tan(zen_v) * 0.01)/ (1/np.cos(zen_v)))**2))**2

        third_term_a = (0.1588 * (-2.5 * np.log10(b_sum/v_sum)))
        fun1 = (v_sum / (b_sum * np.log(10))) * np.sqrt((b_sum_error/b_sum)**2 + (v_sum_error/v_sum)**2)
        third_term_b = np.sqrt((0.079/0.1588)**2 + (fun1/np.log10(b_sum/v_sum))**2)
        third_term_all = (third_term_a * third_term_b)**2

        final_error = np.sqrt(first_term + second_term + third_term_all + 0.742**2)

        return final_error

    def colour_mags_to_txt(self, text_files) -> None:
        for duel_files in text_files:
            exposure = self.b_v_finder(duel_files[0], duel_files[1])
            self.plot_it(exposure)

        with open(FOLDER + f"B-V_{exposure}s.txt", "w") as file:
            file.write("B \t V \t B-V \n")

            for line in self.colour_mags:
                file.write(f"{np.round(line[0], 6)} \t "
                f"{np.round(line[1], 6)} \t "
                f"{np.round(line[2], 6)} \n")

    def plot_it(self, exposure) -> None:
        v_list = [item[1] for item in self.colour_mags]
        b_minus_v = [item[2] for item in self.colour_mags]
        v_error = [item[3] for item in self.colour_mags]
        b_minus_v_error = [item[4] for item in self.colour_mags]

        print(f" error for {exposure} is: {b_minus_v_error}")
        print(f" error for {exposure} is: {v_error}")

        ## Finds the index of min value and pops it, its an outlier
        val, idx = max((val, idx) for (idx, val) in enumerate(b_minus_v))
        b_minus_v.pop(idx)
        v_list.pop(idx)
        v_error.pop(idx)
        b_minus_v_error.pop(idx)

        plt.plot(b_minus_v, v_list, ".", c="black")
        eb1 = plt.errorbar(b_minus_v, v_list, yerr=v_error, linestyle="None")
        eb1[-1][0].set_linestyle(':')

        eb2 = plt.errorbar(b_minus_v, v_list, xerr=b_minus_v_error, linestyle="None", ecolor="r")
        eb2[-1][0].set_linestyle(':')
        plt.title(f"B-V index for {exposure} seconds")
        plt.xlabel("B-V")
        plt.ylabel("V")
        ax = plt.gca()
        ax.invert_yaxis()
        plt.savefig(f"B-V_for_{exposure}s.png")
        plt.show()