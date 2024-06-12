from PIL import Image
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import copy
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import zoom


def spiral_coords(xstart, ystart, arraysize):
    x = y = 0.0
    dx = 1.0
    dy = 0.0
    xloc = np.zeros(arraysize**2)
    yloc = np.zeros(arraysize**2)

    for i in range(arraysize**2):
        xloc[i] = x + xstart
        yloc[i] = y + ystart
        x, y = x + dx, y + dy

        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
            dx, dy = -dy, dx
    return xloc, yloc


def plot_dataset_spiral(images, numImages, bitDepth="8bit",figsize=(7.2,7.2)):
    # Generate spiral indexing
    center = (numImages - 1) // 2
    x, y = spiral_coords(center, center, numImages)
    y = (numImages - 1) - y  # Flip y-axis
    # print(x,y)

    fig, axs = plt.subplots(
        numImages, numImages, figsize=figsize, gridspec_kw={"hspace": 0, "wspace": 0}
    )

    if bitDepth == "float":
        vmax = np.max(images)
        print(vmax)

    for n in range(numImages**2):
        # i = int(x[n])
        # j = int(y[n])

        i = int(y[n])
        j = int(x[n])

        match bitDepth:
            case "8bit":
                axs[i, j].imshow(images[:, :, n], cmap="gray", vmin=0, vmax=255)
            case "10bit":
                axs[i, j].imshow(images[:, :, n], cmap="gray", vmin=0, vmax=1024)
            case "float":
                axs[i, j].imshow(images[:, :, n], cmap="gray", vmin=0, vmax=vmax)
            case "scale":
                axs[i, j].imshow(images[:, :, n], cmap="gray")
        axs[i, j].axis("off")


class ImageLoader:
    def __init__(self, sampleImageFN, minDim, ROI=None):
        sampleImage = np.asarray(Image.open(sampleImageFN))
        ml, nl = sampleImage.shape
        self.ml = ml
        self.nl = nl
        self.minDim = minDim
        self.is8bit = sampleImage.dtype == np.uint8
        self.ROI = ROI

    def get_center(self):
        """Returns center ROI with dimensions minDim x minDim"""
        dm = int((self.ml - self.minDim) / 2)
        dn = int((self.nl - self.minDim) / 2)
        return np.s_[dm : self.ml - dm, dn : self.nl - dn]

    def load_image(self, filename):
        if self.ROI:
            a, b = self.ROI
            roi = np.s_[a : a + self.minDim, b : b + self.minDim]
        else:
            roi = self.get_center()  # get image roi
        image = np.asarray(Image.open(filename))[roi]
        if not self.is8bit:
            image = np.right_shift(image, 6)
        return image

    def load_all_images(self, filenames):
        nImages = len(filenames)
        imStack = np.zeros((self.minDim, self.minDim, nImages))
        for i in range(nImages):
            im = self.load_image(filenames[i])
            imStack[:, :, i] = im
        return imStack


def generate_gaussian_CTF(size, sigma, radius):
    """Generate a 2D Gaussian aperture/filter.

    Parameters:
    - size: tuple of int, the size of the output 2D array (rows, cols).
    - sigma: float, standard deviation of the Gaussian.

    Returns:
    - 2D numpy array representing the Gaussian aperture.
    """
    ylim, xlim = size[1] / 2, size[0] / 2
    x = np.arange(-xlim, xlim, 1)
    y = np.arange(-ylim, ylim, 1)
    x, y = np.meshgrid(x, y)

    gaussian = np.exp(-((x**2 / (2 * sigma**2)) + (y**2 / (2 * sigma**2))))
    cutoff = (x**2 + y**2) < radius**2
    return gaussian * cutoff


class FPMReconstruction:
    """Basic FPM reconstruction using the alternating backprojection algorithm"""

    def __init__(
        self,
        images,
        bitDepth,
        NA,
        LEDsize,
        LEDgap,
        LEDheight,
        z,
        wavelength,
        upsampRatio,
        pixelSize,
        xint=0,
        yint=0,
        apodized=False,
        sigma=10,
        subsampled=False,
    ):
        """
        Input:
        - images (ndarray[m,n,N]): N Low-resolution captured images
        - NA (float): Numerical aperture of objective
        - LEDsize (int): Side length of an odd square matrix ex. 15->15x15
        - LEDheight (float): Height of LED matrix from the sample (in mm)
        - wavelength (float): Wavelength of light illuminating the sample (in m)
        - upsampRatio (int): Upsampling Ratio
        - pixelSize (float): Pixel size of the low resolution camera sensor (in m)
        - xint (int, optional): offset of matrix along x (in mm)
        - yint (int, optional): offset of matrix along y (in mm)
        """

        # Image parameters
        self.images = images
        self.ml, self.nl, self.n_images = images.shape
        self.upsampRatio = upsampRatio
        self.m = self.ml * upsampRatio  # High resolution x
        self.n = self.nl * upsampRatio  # High resolution y
        self.outputs = []
        self.bitDepth = bitDepth

        # Setup parameters
        self.NA = NA
        self.LEDsize = LEDsize
        self.LEDgap = LEDgap
        self.LEDheight = LEDheight
        self.subsampled = subsampled

        # Calculate wavevectors
        xloc, yloc = spiral_coords(xint, yint, LEDsize)
        kxRel = -np.sin(np.arctan(xloc[: self.n_images] * LEDgap / LEDheight))
        kyRel = -np.sin(np.arctan(yloc[: self.n_images] * LEDgap / LEDheight))
        k0 = 2 * np.pi / wavelength
        self.kx = k0 * kxRel
        self.ky = k0 * kyRel
        self.dkx = 2 * np.pi / (pixelSize * self.nl)
        self.dky = 2 * np.pi / (pixelSize * self.ml)

        # Generate Low-pass filter Coherent Transfer Function
        if apodized:
            self.CTF = generate_gaussian_CTF(
                (self.nl, self.ml), sigma, radius=self.nl / 2
            )
            H1 = 1+0j
        else:
            cutoffFrequency = NA * k0
            kmax = np.pi / pixelSize
            print(kmax)
            print(cutoffFrequency)
            XX = np.linspace(-kmax, kmax, int(self.nl))
            YY = np.linspace(-kmax, kmax, int(self.ml))
            kxm, kym = np.meshgrid(XX, YY)
            kzm = np.sqrt(k0**2 - kxm**2 - kym**2)
            # Coherent Transfer Function
            self.CTF = ((kxm**2 + kym**2) < cutoffFrequency**2)
            # Pupil Function
            H1 = np.exp(1j*z*np.real(kzm))*np.exp(-np.abs(z)*np.abs(np.imag(kzm)))
        self.pupil = self.CTF*H1
            

    def reconstruct(self, loops, title, save=True, subsampled=False, method="Basic"):
        # Basic Reconstruction

        # Initial Guess of the object
        objectRecover = zoom(self.images[:,:,0],4)
        # objectRecover = np.ones((int(self.m), int(self.n)))
        objectRecoverFT = ifftshift(ifft2(objectRecover))
        convergence = np.zeros(loops)

        for iteration in range(loops):
            for i in range(self.n_images):
                kxc = np.round((self.n + 1) / 2 + self.kx[i] / self.dkx)
                kyc = np.round((self.m + 1) / 2 - self.ky[i] / self.dky)

                kyl = int(np.floor(kyc - (self.ml / 2)))
                kyh = int(np.floor(kyc + (self.ml / 2)))
                kxl = int(np.floor(kxc - (self.nl / 2)))
                kxh = int(np.floor(kxc + (self.nl / 2)))

                # Subsampled FPM
                # if subsampled:
                #     im_lowCorr = self.upsampRatio**2 * self.images[::2, ::2, i]
                #     im_lowRes[::2, ::2] = im_lowCorr * np.exp(
                #         1j * np.angle(im_lowRes[::2, ::2])
                #     )
                # else:
                #     im_lowCorr = self.upsampRatio**2 * self.images[:, :, i]
                #     im_lowRes = im_lowCorr * np.exp(1j * np.angle(im_lowRes))

                match method:
                    case "Basic":
                        # Basic FPM
                        lowResFT_1 = objectRecoverFT[kyl:kyh, kxl:kxh] * self.CTF
                        im_lowRes = ifft2(ifftshift(lowResFT_1))

                        im_lowCorr = self.upsampRatio**2 * self.images[:, :, i]
                        im_lowRes = im_lowCorr * np.exp(1j * np.angle(im_lowRes))

                        lowResFT_2 = fftshift(fft2(im_lowRes)) * self.CTF
                        objectRecoverFT[kyl:kyh, kxl:kxh] = (
                            1 - self.CTF
                        ) * objectRecoverFT[kyl:kyh, kxl:kxh] + lowResFT_2
                    case "EPRY":
                        # EPRY FPM
                        lowResFT_1 = objectRecoverFT[kyl:kyh, kxl:kxh] * self.pupil
                        im_lowRes = ifft2(ifftshift(lowResFT_1))

                        convergence[iteration] += np.mean(np.abs(im_lowRes))/(
                            np.sum(np.abs(im_lowRes-self.images[:,:,i]))
                            )

                        im_lowCorr = self.upsampRatio**2 * self.images[:, :, i]
                        im_lowRes = im_lowCorr * np.exp(1j * np.angle(im_lowRes))

                        lowResFT_2 = fftshift(fft2(im_lowRes))
                        objectRecoverFT[kyl:kyh, kxl:kxh] = objectRecoverFT[
                            kyl:kyh, kxl:kxh
                        ] + np.conj(self.pupil) / (
                            np.max(np.abs(self.pupil) ** 2)
                        ) * (
                            lowResFT_2 - lowResFT_1
                        )
                        self.pupil += (
                            np.conj(objectRecoverFT[kyl:kyh, kxl:kxh])
                            / (np.max(np.abs(objectRecoverFT[kyl:kyh, kxl:kxh]) ** 2))
                            * (lowResFT_2 - lowResFT_1)
                        )

                if save:
                    self.outputs.append(copy.deepcopy(np.log(abs(objectRecoverFT))))

        objectRecover = ifft2(ifftshift(objectRecoverFT))

        self.plot_results(objectRecover, objectRecoverFT, title, bitDepth=self.bitDepth)
        # print(f'convergence: {convergence}')
        return objectRecover, objectRecoverFT, self.pupil, convergence

    def animate(self, interval=1):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        plt.title("Recovered Spectrum ")
        plt.xlabel("$k_x$")
        plt.ylabel("$k_y$")
        # Allows playing the animation inline in Jupyter notebooks
        plt.rc("animation", html="jshtml")

        plot = ax.imshow(np.log(abs(self.outputs[0])), cmap="gray")

        outputs = self.outputs[:500:interval]

        def step(i):
            plot.set_data(np.log(abs(outputs[i])))
            return plot

        ani = animation.FuncAnimation(
            fig, step, range(len(outputs)), interval=100, blit=False
        )
        return ani

    def plot_results(self, reconstructedImage, reconstructedFT, title, bitDepth=8):
        fig, axs = plt.subplots(1, 4, figsize=(22, 5),dpi=200)
        plt.tight_layout()

        img_vmax = 2**bitDepth

        im0 = axs[0].imshow(self.images[:, :, 0], cmap="gray", vmin=0, vmax=img_vmax)
        axs[0].set_title("Original Image")

        scalebar = ScaleBar(2.48, "um", length_fraction=0.2, location='lower right')  # 1 pixel = 1 micron
        axs[0].add_artist(scalebar)

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax=cax, orientation="vertical")

        im1 = axs[1].imshow(
            np.abs(reconstructedImage), cmap="gray", vmin=0, vmax=img_vmax
        )
        axs[1].set_title("Recovered Intensity")
        
        scalebar = ScaleBar(2.48/4, "um", length_fraction=0.2, location='lower right')  # 1 pixel = 1 micron
        axs[1].add_artist(scalebar)

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")

        im2 = axs[2].imshow(np.angle(reconstructedImage), cmap="gray")
        axs[2].set_title("Recovered Phase")

        scalebar = ScaleBar(2.48/4, "um", length_fraction=0.2, location='lower right')  # 1 pixel = 1 micron
        axs[2].add_artist(scalebar)

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(
            im2,
            cax=cax,
            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            format=mticker.FixedFormatter(
                ["-$\pi$", r"-$\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", "$\pi$"]
            ),
            orientation="vertical",
        )

        im3 = axs[3].imshow(np.log(np.abs(reconstructedFT)), cmap="gray")
        axs[3].set_title("Recovered Spectrum")

        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax, orientation="vertical")

        fig.suptitle(title,y=1.05)
        plt.show()


class FPMSimulation:
    """Simulates FPM acquisition using a low-pass filter"""

    def __init__(
        self,
        image,
        NA,
        LEDsize,
        LEDgap,
        LEDheight,
        wavelength,
        upsampRatio,
        pixelSize,
        bitDepth=8,
        xint=0,
        yint=0,
    ):
        """
        Input:
        image (ndarray[m,n]): Complex Image
        NA (float): Numerical aperture of objective
        LEDsize (int): Side length of an odd square matrix ex. 15->15x15
        LEDheight (float): Height of LED matrix from the sample (in mm)
        wavelength (float): Wavelength of light illuminating the sample (in m)
        upsampRatio (int): Upsampling Ratio
        pixelSize (float): Pixel size of the low resolution camera sensor (in m)
        bitDepth (int): number of bits of the image
        xint (int, optional): offset of matrix along x (in mm)
        yint (int, optional): offset of matrix along y (in mm)
        """

        # Image parameters
        self.image = image
        self.m, self.n = image.shape
        self.ml = self.m // upsampRatio  # Low resolution x
        self.nl = self.n // upsampRatio  # Low resolution y
        self.images = np.zeros((self.ml, self.nl, LEDsize**2))

        # Setup parameters
        self.NA = NA
        self.LEDsize = LEDsize
        self.LEDgap = LEDgap
        self.LEDheight = LEDheight

        # Calculate wavevectors
        xloc, yloc = spiral_coords(xint, yint, LEDsize)
        kxRel = -np.sin(np.arctan(xloc * LEDgap / LEDheight))
        kyRel = -np.sin(np.arctan(yloc * LEDgap / LEDheight))
        k0 = 2 * np.pi / wavelength
        self.kx = k0 * kxRel
        self.ky = k0 * kyRel
        self.dkx = 2 * np.pi / (pixelSize * self.nl)
        self.dky = 2 * np.pi / (pixelSize * self.ml)

        # Generate Low-pass filter Coherent Transfer Function
        cutoffFrequency = NA * k0
        kmax = np.pi / pixelSize
        XX = np.linspace(-kmax, kmax, int(self.nl))
        YY = np.linspace(-kmax, kmax, int(self.ml))
        kxm, kym = np.meshgrid(XX, YY)
        # Coherent Transfer Function
        self.CTF = (kxm**2 + kym**2) < cutoffFrequency**2

    def simulate(self, show=True):
        complexObjectFT = fftshift(fft2(self.image))

        for i in range(self.LEDsize**2):
            kxc = np.round((self.n / 2) + self.kx[i] / self.dkx)
            kyc = np.round((self.m / 2) - self.ky[i] / self.dky)

            kyl = int(np.round(kyc - (self.ml / 2)))
            kyh = int(np.round(kyc + (self.ml / 2)))
            kxl = int(np.round(kxc - (self.nl / 2)))
            kxh = int(np.round(kxc + (self.nl / 2)))

            imSeqLowFT = (
                (self.m / self.ml) ** 2 * complexObjectFT[kyl:kyh, kxl:kxh] * self.CTF
            )
            self.images[:, :, i] = abs(ifft2(ifftshift(imSeqLowFT)))

        # Discretize Image
        self.images = np.round((self.images / self.images.max()) * 255)

        if show:
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))
            start = 0
            for i in range(5):
                axs[i].imshow(self.images[:, :, i], cmap="gray")
                axs[i].set_title(f"$i={start+i}$")

        return self.images
