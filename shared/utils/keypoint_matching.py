"""Implements keypoint matching for a pair of images."""
import os
import numpy as np
import PIL
import cv2
import matplotlib.pyplot as plt


def show_single_image(img, figsize=(7, 5), title="Single image"):
    """Displays a single image."""
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(img)
    plt.title(title)
    plt.show()


def show_two_images(img1, img2, title="Two images"):
    """Displays a pair of images."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    ax[0].axis("off")
    ax[0].imshow(img1)

    ax[1].axis("off")
    ax[1].imshow(img2)

    plt.suptitle(title)
    plt.show()


def show_three_images(img1, img2, img3, ax1_title="", ax2_title="", ax3_title="", title="Three images"):
    """Displays a triplet of images."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    ax[0].axis("off")
    ax[0].imshow(img1)
    ax[0].set_title(ax1_title)

    ax[1].axis("off")
    ax[1].imshow(img2)
    ax[1].set_title(ax2_title)

    ax[2].axis("off")
    ax[2].imshow(img3)
    ax[2].set_title(ax3_title)

    plt.suptitle(title)
    plt.show()


class KeypointMatcher:
    """Class for Keypoint matching for a pair of images."""

    def __init__(self, **sift_args) -> None:
        self.SIFT = cv2.SIFT_create(**sift_args)
        self.BFMatcher = cv2.BFMatcher()
    
    @staticmethod
    def _check_images(img1: np.ndarray, img2: np.ndarray):
        assert isinstance(img1, np.ndarray)
        assert len(img1.shape) == 2

        assert isinstance(img2, np.ndarray)
        assert len(img2.shape) == 2

        # assert img1.shape == img2.shape
    
    @staticmethod
    def _show_matches(img1, kp1, img2, kp2, matches, K=10, figsize=(10, 5), drawMatches_args=dict(matchesThickness=3, singlePointColor=(0, 0, 0))):
        """Displays matches found in the image"""
        selected_matches = np.random.choice(matches, K)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, outImg=None, **drawMatches_args)
        show_single_image(img3, figsize=figsize, title=f"Randomly selected K = {K} matches between the pair of images.")
        return img3

    def match(self, img1: PIL.Image, img2: PIL.Image, show_matches: bool = True):
        """Finds, describes and matches keypoints in given pair of images."""
        
        img1 = np.array(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        
        img2 = np.array(img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # check input images
        self._check_images(img1, img2)

        # find kps and descriptors in each image
        kp1, des1 = self.SIFT.detectAndCompute(img1, None)
        kp2, des2 = self.SIFT.detectAndCompute(img2, None)

        # compute matches via Brute-force matching
        matches = self.BFMatcher.match(des1, des2)

        # sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)

        if show_matches:
            self._show_matches(img1, kp1, img2, kp2, matches)

        return matches, kp1, des1, kp2, des2


def warp(im, M, output_shape):
    out = np.zeros((output_shape[0], output_shape[1]))
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            u, v = np.array([[i, j, 0, 0, 1, 0], [0, 0, i, j, 0, 1]]) @ M
            u = int(round(u))
            v = int(round(v))
            if im.shape[0] > u >= 0 and im.shape[1] > v >= 0:
                out[i, j] = im[u, v]

    return out


def project_2d_to_6d(X: np.ndarray):
    """Projects X (N x 2) to Z (2N x 6) space."""
    N = len(X)
    assert X.shape == (N, 2)

    Z = np.zeros((2 * N, 6))
    # in columns 0 to 2, fill even indexed rows of Z with X, and fill 5th column with 1
    Z[::2, 0:2] = X
    Z[::2, 4] = 1.0
    # in columns 2 to 4, fill odd indexed rows of Z with X
    Z[1::2, 2:4] = X
    Z[1::2, 5] = 1.0

    return Z


def project_6d_to_2d(Z: np.ndarray):
    """Projects Z (2N x 6) to X (N x 2) space."""
    N = len(Z) // 2
    assert Z.shape == (2 * N, 6)

    X_from_even_rows = Z[::2, 0:2]
    X_from_odd_rows = Z[1::2, 2:4]
    assert (X_from_even_rows == X_from_odd_rows).all()

    return X_from_even_rows



def project_2d_to_1d(X: np.ndarray):
    """Returns X (N x 2) from Z (2N, 1)"""
    N = len(X)
    X_stretched = np.zeros(2 * N)
    X_stretched[::2] = X[:, 0]
    X_stretched[1::2] = X[:, 1]
    return X_stretched


def project_1d_to_2d(Z: np.ndarray):
    """Returns X (N x 2) from Z (2N, 1)"""
    N = len(Z) // 2
    assert Z.shape == (2 * N,)

    X = np.zeros((N, 2))
    X[:, 0] = Z[::2]
    X[:, 1] = Z[1::2]

    return X


def rigid_body_transform(X: np.ndarray, params: np.ndarray):
    """Performs rigid body transformation of points X (N x 2) using params (6 x 1 flattened)"""
    N = len(X)
    assert X.shape == (N, 2)

    X = project_2d_to_6d(X)

    X_transformed = np.matmul(X, params)
    X_transformed = project_1d_to_2d(X_transformed)
    assert X_transformed.shape == (N, 2)

    return X_transformed


def rigid_body_transform_params(X1: np.ndarray, X2: np.ndarray):
    """Returns rigid-body transform parameters RT (6 x 1) assuming transformation between X1 and X2"""
    N = len(X1)
    assert X1.shape == X2.shape
    assert X1.shape == (N, 2)

    # X2 = X1 * params => params = psuedoinverse(X1) * X2
    X1_expanded = project_2d_to_6d(X1)
    assert X1_expanded.shape == (2 * N, 6)

    X2_stretched = project_2d_to_1d(X2)
    assert X2_stretched.shape == (2 * N,)

    params = np.dot(np.linalg.pinv(X1_expanded), X2_stretched)
    return params


class ImageAlignment:
    """Class to perform alignment of a pair of images given keypoints."""

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def show_transformed_points(img1, img2, X1, kp1, kp2, matches, params, num_inliers, num_to_show=20):
        import matplotlib.cm as cm

        H1, W1 = img1.shape
        H2, W2 = img2.shape
        img = np.hstack([img1, img2])

        random_matches = np.random.choice(matches, num_to_show)

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        colors = cm.rainbow(np.linspace(0, 1, num_to_show))

        for i, match in enumerate(random_matches):

            # select a single match to visualize
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt

            # get (x1, y1) transformed to (x1_transformed, y1_transformed)
            A = project_2d_to_6d(np.array([[x1, y1]]))
            (x1_transformed, y1_transformed) = np.dot(A, params)

            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.scatter(x1_transformed + W1, y1_transformed, s=200, marker="x", color=colors[i])
            ax.plot(
                (x1, x1_transformed + W1), (y1, y1_transformed),
                linestyle="--", color=colors[i], marker="o",
            )

        ax.set_title(
            f"Points in image 1 mapped to transformed points estimated by {num_inliers} points.",
            fontsize=18,
        )

        os.makedirs("./results/", exist_ok=True)
        plt.savefig(f"./results/match_transformed_inliers_{num_inliers}.png", bbox_inches="tight")
        plt.show()

    def ransac(
            self, img1, kp1, img2, kp2, matches, num_matches=6, max_iter=500,
            radius_in_px=10, show_transformed=True, inlier_th_for_show=1000
        ):
        """Performs RANSAC to find best matches."""

        best_inlier_count = 0
        best_params = None

        # get coordinates of all points in image 1
        X1 = np.array([kp1[matches[i].queryIdx].pt for i in range(len(matches))])

        # get coordinates of all points in image 2
        X2 = np.array([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

        for i in range(max_iter):
            # choose matches randomly
            selected_matches = np.random.choice(matches, num_matches)

            # get matched keypoints in img1
            X1_selected = np.array([kp1[selected_matches[i].queryIdx].pt for i in range(len(selected_matches))])

            # get matched keypoints in img2
            X2_selected = np.array([kp2[selected_matches[i].trainIdx].pt for i in range(len(selected_matches))])

            # get transformation parameters
            params = rigid_body_transform_params(X1_selected, X2_selected)
            
            # transform X1 to get X2_transformed
            X2_transformed = rigid_body_transform(X1, params)

            # find inliers
            diff = np.linalg.norm(X2_transformed - X2, axis=1)
            indices = diff < radius_in_px
            num_inliers = sum(indices)
            if num_inliers > best_inlier_count:
                print(f"Found {num_inliers} inliers!")
                best_params = params
                best_inlier_count = num_inliers

                if show_transformed and num_inliers > inlier_th_for_show:
                    self.show_transformed_points(img1, img2, X1, kp1, kp2, matches, best_params, num_inliers)

        return best_params
    
    def align(
            self, img1, kp1, img2, kp2, matches, num_matches=6,
            max_iter=500, show_warped_image=True,
            save_warped=False, path="results/sample.png",
            method="custom"
        ):
        best_params = self.ransac(img1, kp1, img2, kp2, matches, max_iter=max_iter, num_matches=num_matches)

        # apply the affine transformation using cv2.warpAffine()
        rows, cols = img1.shape[:2]

        if method == 'custom':
            img1_warped = warp(img1, best_params, (rows, cols))
        else:
            M = np.zeros((2, 3))
            M[0, :2] = best_params[:2]
            M[1, :2] = best_params[2:4]
            M[0, 2] = best_params[4]
            M[1, 2] = best_params[5]
            img1_warped = cv2.warpAffine(img1, M, (cols, rows))

        if show_warped_image:
            show_three_images(
                img1, img2, img1_warped, title="",
                ax1_title="Image 1", ax2_title="Image 2", ax3_title="Transformation: Image 1 to Image 2",
            )

        if save_warped:
            plt.imsave(path, img1_warped)

        return best_params


if __name__ == "__main__":
    # read & show images
    boat1 = cv2.imread('boat1.pgm', cv2.IMREAD_GRAYSCALE)
    boat2 = cv2.imread('boat2.pgm', cv2.IMREAD_GRAYSCALE)
    show_two_images(boat1, boat2, title="Given pair of images.")

    kp_matcher = KeypointMatcher(contrastThreshold=0.1, edgeThreshold=5)
    matches, kp1, des1, kp2, des2 = kp_matcher.match(boat1, boat2, show_matches=True)