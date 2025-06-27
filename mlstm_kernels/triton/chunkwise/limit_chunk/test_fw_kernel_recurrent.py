import unittest
import torch
import triton
import triton.language as tl

# Import the raw Triton kernel from your main source file.
from fw_kernel_recurrent import mlstm_chunkwise__recurrent_fw_C_kernel

# Define the Python launcher function locally in the test script.
def launch_recurrent_fw_c_kernel(matK, matV, vecB, vecI, matC_initial, vecN_initial, scaMinter_initial, use_initial_state=False):
    # This launcher is the same as before. It prepares arguments and launches the imported kernel.
    B, NH, S, DHQK = matK.shape
    _, _, _, DHHV = matV.shape
    _, _, NC, L = vecB.shape
    siz_b_DHQK, siz_b_DHHV = 32, 64
    matC_states = torch.zeros(B, NH, (NC + 1) * DHQK, DHHV, device=matK.device, dtype=torch.float32)
    vecN_states = torch.zeros(B, NH, (NC + 1) * DHQK, device=matK.device, dtype=torch.float32)
    scaMinter_states = torch.zeros(B, NH, (NC + 1), device=matK.device, dtype=torch.float32)
    grid = (triton.cdiv(DHQK, siz_b_DHQK), triton.cdiv(DHHV, siz_b_DHHV), B * NH)
    TORCH_DTYPE_TO_TRITON_DTYPE = {
        torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16,
    }
    triton_dtype = TORCH_DTYPE_TO_TRITON_DTYPE[matV.dtype]
    mlstm_chunkwise__recurrent_fw_C_kernel[grid](
        matK, matV, vecB, vecI, matC_initial, vecN_initial, scaMinter_initial,
        matC_states, vecN_states, scaMinter_states,
        matK.stride(0), matK.stride(2), matK.stride(3), matV.stride(0), matV.stride(2), matV.stride(3),
        vecB.stride(0), vecB.stride(2), vecB.stride(3), matC_states.stride(0), matC_states.stride(2),
        matC_states.stride(3), vecN_states.stride(0), vecN_states.stride(2), scaMinter_states.stride(0),
        scaMinter_states.stride(2), matC_initial.stride(0), matC_initial.stride(2), matC_initial.stride(3),
        vecN_initial.stride(0), vecN_initial.stride(2), scaMinter_initial.stride(0),
        B=B, NH=NH, S=S, DHQK=DHQK, DHHV=DHHV, NC=NC, L=L,
        siz_b_DHQK=siz_b_DHQK, siz_b_DHHV=siz_b_DHHV,
        USE_INITIAL_STATE=use_initial_state, DTYPE=triton_dtype
    )
    return matC_states, vecN_states, scaMinter_states

class TestMlstmRecurrentFwKernel(unittest.TestCase):

    def test_kernel_runs_and_is_numerically_correct(self):
        """
        Tests that the kernel compiles and runs with the correct syntax,
        and that the value from tl.max is numerically correct.
        This test will PASS with the correct code and FAIL with the incorrect code.
        """
        # Arrange: Set up specific, known inputs
        B, NH, S, DHQK, DHHV, NC, L = 1, 1, 16, 64, 128, 1, 16
        device = 'cuda'
        dtype = torch.float32
        known_max_val = 100.0
        vecI_input = torch.ones(B, NH, NC, L, device=device, dtype=dtype)
        vecI_input[0, 0, 0, 7] = known_max_val # vecA_k_val will have a max of 100.0
        vecB_input = torch.zeros(B, NH, NC, L, device=device, dtype=dtype) # Zeros so it doesn't affect vecA
        matK = torch.zeros(B, NH, S, DHQK, device=device, dtype=dtype)
        matV = torch.zeros(B, NH, S, DHHV, device=device, dtype=dtype)
        matC_initial = torch.zeros(B, NH, DHQK, DHHV, device=device, dtype=dtype)
        vecN_initial = torch.zeros(B, NH, DHQK, device=device, dtype=dtype)
        scaMinter_initial = torch.zeros(B, NH, device=device, dtype=dtype)

        # Act & Assert
        try:
            _, _, scaMinter_states = launch_recurrent_fw_c_kernel(
                matK, matV, vecB_input, vecI_input, matC_initial, vecN_initial, scaMinter_initial, use_initial_state=True
            )
            # The logic inside the kernel ensures the final state is equal to the max value found
            final_minter_state = scaMinter_states[0, 0, 1].item()
            self.assertAlmostEqual(final_minter_state, known_max_val, places=4,
                                 msg="Kernel ran, but output is numerically incorrect.")
        except Exception as e:
            # If any exception occurs (like the CompilationError), fail the test.
            self.fail(f"Kernel launch failed unexpectedly with exception: {e}")

if __name__ == '__main__':
    unittest.main()