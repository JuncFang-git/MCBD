# pylint: disable=wildcard-import
import cv2
import logging
import colour
import unittest
import time
import numpy as np
from scipy import ndimage
from scipy.special import expit
from edge_link import *

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

class PU21:
    # ref: https://github.com/gfxdisp/pu21/blob/main/matlab/pu21_encoder.m
    def __init__(self, type="banding_glare") -> None:
        self.L_min = 0.005
        self.L_max = 10000
        self.epsilon = 1e-5
        
        if type == "banding":
            self.par = [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484]
        elif type == "banding_glare":
            self.par = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
        elif type == "peaks":
            self.par = [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577]
        elif type == "peaks_glare":
            self.par = [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
        else:
            logging.error(f"Unknown type: {type}")
        
    def encode(self, Y: np.ndarray) -> np.ndarray:
        """Convert from linear (optical) values Y to encoded (electronic) values V
        
        Args:
            Y (np.ndarray): is in the range from 0.005 to 10000. The values MUST be scaled in the absolute units (nits, cd/m^2).
        Returns:
            V (np.ndarray): is in the range from 0 to circa 600 (depends on the encoding used). 100 [nit] is mapped to 256 to \\
                mimic the input to SDR quality metrics. 
        """

        if np.any(Y < (self.L_min - self.epsilon)) or np.any(Y > self.L_max + self.epsilon):
            logging.warning('Values passed to encode are outside the valid range')
        Y = np.clip(Y, self.L_min, self.L_max)
        V = self.par[6] * (((self.par[0] + self.par[1]*Y**self.par[3]) / (1 + self.par[2]*Y**self.par[3]))**self.par[4] - self.par[5])
        V = np.clip(V, 0, None)
        return V
        
    
    def decode(self, V: np.ndarray) -> np.ndarray:
        """Convert from encoded (electronic) values V into linear (optical) values Y

        Args:
            V (np.ndarray): is in the range from 0 to circa 600.

        Returns:
            Y (np.ndarray): is in the range from 0.005 to 10000
        """
        V_p = np.clip(V / self.par[6] + self.par[5], 0, None) ** (1 / self.par[4])
        Y = (np.clip(V_p - self.par[0], 0, None) / (self.par[1] - self.par[2]*V_p))**(1 / self.par[3])
        return Y    


class MCBD(object):
    def __init__(self) -> None:
        # hyper parameters for edge_strength_clues
        # self.T1, self.T2 = 1, 5 
        self.T1, self.T2 = 1, 50 
        # hyper parameters for length_clues
        # self.E0, self.tau = 132, 4.16
        self.E0, self.tau = 2, 4.16
        # hyper parameters for forward
        self.alpha, self.beta = 0.5, 0.25
    
    def luminance_clues(self, I: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            I (np.ndarray): a numpy array of cv2 image format
        """
        # eotf 
        ## I from [0, 255] to [0., 1.]
        I = I / 255.
        ## PQ curve
        eotf_res =  colour.eotf(I, function='ITU-R BT.2100 PQ')
        ## min value adjust
        ## PQ curve have the min value at 0.005, but colour.eotf output the min value at 0
        eotf_res = np.clip(eotf_res, 0.005, None)
        
        # pu21
        P = PU21().encode(eotf_res)
        return P
    
    def edge_strength_clues(self, P: np.ndarray) -> np.ndarray:
        # edge strength calculation
        sobel_h = abs(ndimage.sobel(P, 0))  # horizontal gradient
        sobel_v = abs(ndimage.sobel(P, 1))  # vertical gradient
        S = np.where(sobel_h > sobel_v, sobel_h, sobel_v)
        # Test.helper_save_as_gray(ndimage.sobel(P, 0), "tmp_sobel_h.png")
        # Test.helper_save_as_gray(ndimage.sobel(P, 1), "tmp_sobel_v.png")
        # Test.helper_save_as_gray(ndimage.sobel(P), "tmp_sobel.png")
        # Test.helper_save_as_gray(S, "tmp_S.png")
        S_binary = np.where(S>0, 255, 0)
        # Test.helper_save_as_gray(S_binary, "tmp_S_binary.png")
        
        # get candidate map
        ## flat check
        F = flat_checker(P) # TODO:make sure
        ## filter
        G = np.where(
            np.logical_and(np.logical_and(S > self.T1, S < self.T2), F), 
            S, 
            0)
        return G, F, S

    def length_clues(self, G: np.ndarray) -> np.ndarray:
        # Test.helper_save_as_gray(G, "tmp_L_G.png")
        # edge_link
        # Edge = edgelink(G, minilength=200) #
        Edge = edgelink(np.where(G>0, 255, 0)) #
        t0 = time.time()
        Edge.get_edgelist()
        logging.debug(f"function `get_edgelist` used time of {time.time() - t0}s ")
        edge_list = Edge.edgelist
        logging.debug(f"num_edge: {len(edge_list)} ")
        # get L
        # TODO: can faster ?
        t0 = time.time()
        L = np.zeros_like(G)
        # debug_edge = np.zeros_like(G)
        # debug_edge2 = np.zeros_like(G)
        # length_static = [0] * 9
        for edge in edge_list:
            edge_len = edge.shape[0]
        #     if edge_len <= 10:
        #         length_static[0] += 1
        #     if edge_len > 10:
        #         length_static[1] += 1
        #     if edge_len > 20:
        #         length_static[2] += 1
        #     if edge_len > 40:
        #         length_static[3] += 1
        #     if edge_len > 80:
        #         length_static[4] += 1
        #     if edge_len > 160:
        #         length_static[5] += 1
        #     if edge_len > 320:
        #         length_static[6] += 1
        #     if edge_len > 500:
        #         length_static[7] += 1
        #     if edge_len > 1000:
        #         length_static[8] += 1
                    
            for point_idx in range(edge_len):
                x, y = edge[point_idx][0], edge[point_idx][1]
                value = expit(edge_len / self.E0 - self.tau)
                L[x, y] = value if value > L[x, y] else L[x, y] # use the max length 
        #         debug_edge[x, y] = 255
        #         debug_edge2[x, y] = 255 if edge_len > 10 else 0
        # logging.debug(f"debug_length_static: {length_static} ")
        # logging.debug(f"for loop for get L used time of {time.time() - t0}s ")
        # Test.helper_save_as_gray(L, "tmp_L_L.png")
        # Test.helper_save_as_gray(debug_edge, "tmp_L_debug_edge.png")
        # Test.helper_save_as_gray(debug_edge2, "tmp_L_debug_edge2.png")
        # Test.helper_save_as_gray(debug_edge2*G, "tmp_L_debug_res.png")
        return L * G
    
    def single_forward(self, img:np.ndarray):
        return self.length_clues(self.edge_strength_clues(self.luminance_clues(img))[0])
    
    def __call__(self, img:np.ndarray):
        """_summary_

        Args:
            img (np.ndarray): input 1-channel image, could be grayscale, or Y-plane of YUV

        Returns:
            bvm(np.ndarray): banding visual map
            q(float): banding index (score)
        """
        b0 = self.single_forward(img) # input scale
        b1 = self.single_forward(ndimage.zoom(img, 0.5)) # 1/2 scale
        b2 = self.single_forward(ndimage.zoom(img, 0.25)) # 1/4 scale
        
        bvm = b0 + self.alpha*ndimage.zoom(b1, 2) + self.beta*ndimage.zoom(b2, 4)
        q = np.mean(bvm)
        return bvm, q
    
def filter_k3_func1(inp:np.array, out:np.array):
    """Calling this function at each line of `generic_filter1d` input. 
    The arguments of the line are the input line, and the output line. 
    The input and output lines are 1-D double arrays. 
    The input line is extended appropriately according to the filter size and origin. 
    The output line must be modified in-place with the result.

    Args:
        inp (np.array): 1-D array, is extended appropriately according to the filter size and origin.  
        out (np.array): 1-D array.

    """
    kernel = np.array([-1, 0, 1])
    view = np.lib.stride_tricks.sliding_window_view(inp, len(kernel))
    out[:] = np.sum(view * kernel, axis=-1)
    
def filter_k5_func2(inp:np.array, out:np.array):
    """same as filter_k3_func1 
    """
    kernel = np.array([-1, 0, 0, 0, 1])
    view = np.lib.stride_tricks.sliding_window_view(inp, len(kernel))
    out[:] = np.sum(view * kernel, axis=-1)
    
def filter_k7_func3(inp:np.array, out:np.array):
    """same as filter_k3_func1 
    """
    kernel = np.array([-1, 0, 0, 0, 0, 0, 1])
    view = np.lib.stride_tricks.sliding_window_view(inp, len(kernel))
    out[:] = np.sum(view * kernel, axis=-1)
    
def flat_checker(P: np.ndarray) -> np.ndarray:
    # hyper parameters (IS VERY IMPORTANT)
    lambda_sobel = 20 
    lambda0 = 10 
    gaus_sigma = 100
    gaus_radius = 50
    
    # get S_x, S_y and combine them.
    S_x1 = ndimage.generic_filter1d(P, filter_k3_func1, 3, axis=1)
    S_x2 = ndimage.generic_filter1d(P, filter_k5_func2, 5, axis=1)
    S_x3 = ndimage.generic_filter1d(P, filter_k7_func3, 7, axis=1)
    
    # S_x = np.where(abs(S*S_x2 + S*S_x3 - 2*S*S_x1)<1, S*S_x1, S)
    # S_x = np.where(abs(S_x2 + S_x3 - 2*S_x1)<lambda_sobel, 1, 0) # I adjust this logical to genrate binary map
    S_x = np.where(abs(S_x2 + S_x3 - 2*S_x1)<lambda_sobel, S_x1, S_x1.min()) # I adjust this logical to genrate binary map
    
    # Test.helper_save_as_gray(S_x1, "tmp_Sx1.png")
    # Test.helper_save_as_gray(S_x2, "tmp_Sx2.png")
    # Test.helper_save_as_gray(S_x3, "tmp_Sx3.png")
    # Test.helper_save_as_gray(S_x, "tmp_Sx.png")
    # S_x_b = np.where(S_x>0, 1, 0)
    # Test.helper_save_as_gray(S_x_b, "tmp_Sx_b.png")
    
    S_y1 = ndimage.generic_filter1d(P, filter_k3_func1, 3, axis=0)
    S_y2 = ndimage.generic_filter1d(P, filter_k5_func2, 5, axis=0)
    S_y3 = ndimage.generic_filter1d(P, filter_k7_func3, 7, axis=0)
    
    # S_y = np.where(abs(S*S_y2 + S*S_y3 - 2*S*S_y1)<1, S*S_y1, S)
    # S_y = np.where(abs(S_y2 + S_y3 - 2*S_y1)<lambda_sobel, 1, 0) # I adjust this logical to genrate binary map
    S_y = np.where(abs(S_y2 + S_y3 - 2*S_y1)<lambda_sobel, S_y1, S_y1.min()) # I adjust this logical to genrate binary map
    # Test.helper_save_as_gray(S_y1, "tmp_Sy1.png")
    # Test.helper_save_as_gray(S_y2, "tmp_Sy2.png")
    # Test.helper_save_as_gray(S_y3, "tmp_Sy3.png")
    # Test.helper_save_as_gray(S_x, "tmp_Sy.png")
    
    S_sobel = np.sqrt(S_x**2 + S_y**2) # sobel combine (not clear at paper)
    # Test.helper_save_as_gray(S_sobel, "tmp_S_sobel.png")
    # S_sobel_b = np.where(S_sobel>0, 1, 0)
    # Test.helper_save_as_gray(S_sobel_b, "tmp_S_sobel_b.png")
    
    # Gaussian 
    S_Gaus = ndimage.gaussian_filter(S_sobel, sigma=gaus_sigma, radius=gaus_radius) # params is casually. (not clear at paper)
    # Test.helper_save_as_gray(S_Gaus, "tmp_S_Gaus.png")
    
    # filter
    F = np.where(abs(S_Gaus - S_sobel) > lambda0, 0, 1)
    return F
    
    
class Test(unittest.TestCase):
    mcbd = MCBD()
    mcbd_example = cv2.imread("test_input/mcbd.png")
    mcbd_example_gray = cv2.cvtColor(mcbd_example, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("test_output/mcbd_gray.png", mcbd_example_gray)
    
    edge_link_example = cv2.imread("test_input/edge_link.png")
    edge_link_example_gray = cv2.cvtColor(edge_link_example, cv2.COLOR_BGR2GRAY)
    @classmethod
    def helper_save_as_gray(cls, arr:np.ndarray, file_path:str):
        """save a 1D array as a gray image by the cv2

        Args:
            arr (np.ndarray): 1D array
            file_path (str): save path (path/filename)
        """
        # normal
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.
        arr = np.array(arr, dtype=np.uint8)
        
        # save
        cv2.imwrite(file_path, arr)
        
    def _test_pu21(self):
        pu21 = PU21()
        test_example = np.array([[1.32340404,  1.87238002,  1.750423],
                                 [2.75338878,  3.7186751 ,  4.17476571]])
        self.assertIsNone(
            np.testing.assert_almost_equal(
                test_example, pu21.decode(pu21.encode(test_example))
                )
            )
        # this resut from referenced matlab code
        self.assertIsNone(
            np.testing.assert_almost_equal(
                pu21.encode(test_example),
                [[43.9277, 54.3851, 52.2420],
                [67.6844, 79.2211, 83.9225]],
                decimal=4
                )
            )
        
    def _test_edge_link(self):
        Edge = edgelink(self.edge_link_example_gray)                               # 200 means nothing here. Don't mind it.
        t0 = time.time()
        Edge.get_edgelist()
        # self.assertTrue(((time.time() - t0)) < 3 * 2.74 ) # 2.74s at matlab
        print(f"Time: P-{time.time() - t0}s, M-2.74s")
        edgelist = Edge.edgelist
        num_edge = len(edgelist)
        # self.assertTrue(num_edge == 4512) # 4512 edges detected at matlab
        print(f"NumEdge: P-{num_edge}, M-4512")
        length_static = [0] * 9
        for edge in edgelist:
            edge_len = edge.shape[0]
            if edge_len <= 10:
                length_static[0] += 1
            if edge_len > 10:
                length_static[1] += 1
            if edge_len > 20:
                length_static[2] += 1
            if edge_len > 40:
                length_static[3] += 1
            if edge_len > 80:
                length_static[4] += 1
            if edge_len > 160:
                length_static[5] += 1
            if edge_len > 320:
                length_static[6] += 1
            if edge_len > 500:
                length_static[7] += 1
            if edge_len > 1000:
                length_static[8] += 1
        # self.assertEqual(
        #     np.all(
        #         np.array(length_static)==\
        #             np.array([3334,1178,758,522,321,163,56,18]) # static at matlab
        #             )
        #     )
        print(f"LenStatic: \nP-{length_static}, \nM-[3334,1178,758,522,321,163,56,18,1]")
       
    def _test_luminance_clues(self):
        res = self.mcbd.luminance_clues(self.mcbd_example_gray)
        self.helper_save_as_gray(res, "test_output/luminance.png")
    
    def _test_edge_strength_clues(self):
        G, F, S = self.mcbd.edge_strength_clues(
            self.mcbd.luminance_clues(self.mcbd_example_gray)
        )
        self.helper_save_as_gray(G, "test_output/edge_strength.png")
        # self.helper_save_as_gray(np.where(G>G.min(), 255, 0), "test_output/edge_strength_binary.png")
        self.helper_save_as_gray(F, "test_output/edge_strength_F.png")
        self.helper_save_as_gray(S, "test_output/edge_strength_S.png")
        
    def _test_length_clues(self):
        res = self.mcbd.single_forward(self.mcbd_example_gray)
        self.helper_save_as_gray(res, "test_output/length.png")
        
    def test_mcbd(self):
        bvm, bs = self.mcbd(self.mcbd_example_gray)
        self.helper_save_as_gray(bvm, "test_output/mcbd_res.png")
        print(f"banding_score:{bs}")
        self.assertIsNotNone(bvm)
        self.assertIsNotNone(bs)

if __name__ == "__main__":
    unittest.main()
    pass
