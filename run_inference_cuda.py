"""
ORT C API 加载模型 + 单条请求推理。

用法（在 19923 上）:
  cd /export/chenxuan/ort_test && source env.sh && python3 run_inference.py
"""

import ctypes
import os
import time
import numpy as np

# ── 路径配置 ──────────────────────────────────────────────
CUDA_LIB  = "/usr/local/cuda/lib64"
ORT_LIB   = "/App/predictor/lib"
ONNX_PATH = "/export/chenxuan/ort_test/optimized_model.onnx"
REQ_DIR   = "/export/chenxuan/ort_test/req_799_npz"

os.environ["LD_LIBRARY_PATH"] = (
    f"{CUDA_LIB}:{ORT_LIB}:" + os.environ.get("LD_LIBRARY_PATH", "")
)

# ── ORT C API 索引（v18，从 ort_107 header 逐成员计数验证）────
# 验证基准：GetErrorMessage=2, CreateEnv=3, CreateSession=7,
#            CreateSessionOptions=10, SetGraphOptimizationLevel=23
I_GetErrorMessage            = 2
I_CreateSession              = 7
I_Run                        = 9
I_CreateSessOpts             = 10
I_SetExecMode                = 13   # SetSessionExecutionMode
I_DisableCpuMemArena         = 19
I_SetGraphOptLevel           = 23
I_SetIntraOpNumThreads       = 24
I_SetInterOpNumThreads       = 25
I_SessionGetInputCount       = 30
I_SessionGetOutputCount      = 31
I_SessionGetInputName        = 36
I_SessionGetOutputName       = 37
I_CreateTensorWithData       = 49
I_GetTensorMutableData       = 51
I_GetTensorElementType       = 60
I_GetDimensionsCount         = 61
I_GetDimensions              = 62
I_GetTensorTypeAndShape      = 65
I_CreateCpuMemoryInfo        = 69
I_AllocatorFree              = 76
I_GetAllocatorWithDefault    = 78
I_ReleaseValue               = 96
I_ReleaseTensorTypeAndShape  = 99
I_DisablePerSessionThreads   = 120
I_CreateEnvWithGlobalThreadPools = 119
I_CreateThreadingOptions     = 121
I_ReleaseThreadingOptions    = 122
I_SetGlobalIntraOpNumThreads = 147
I_SetGlobalInterOpNumThreads = 148
I_AddSessionConfigEntry      = 130
I_AppendExecutionProvider_CUDA_V2 = 204
I_CreateCUDAProviderOptions  = 205
I_UpdateCUDAProviderOptions  = 206
I_ReleaseCUDAProviderOptions = 208

# numpy dtype → ONNX element type 映射见 numpy_to_ort_value()

VP = ctypes.c_void_p
PS = ctypes.sizeof(VP)
_G = None
_ort_so = None


def _get_fn(idx):
    return ctypes.cast(_G + idx * PS, ctypes.POINTER(VP)).contents.value


def _chk(st):
    if st:
        fn = ctypes.cast(_get_fn(I_GetErrorMessage),
                         ctypes.CFUNCTYPE(ctypes.c_char_p, VP))
        raise RuntimeError(f"ORT: {fn(st).decode()}")


def init_ort():
    global _G, _ort_so
    for lib in [
        f"{CUDA_LIB}/libcudart.so.12",
        f"{CUDA_LIB}/libcublas.so.12",
        f"{CUDA_LIB}/libcublasLt.so.12",
        f"{ORT_LIB}/libonnxruntime_providers_shared.so",
        f"{ORT_LIB}/libonnxruntime.so.1.18.0",
    ]:
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    _ort_so = ctypes.CDLL(f"{ORT_LIB}/libonnxruntime.so.1.18.0")

    class OrtApiBase(ctypes.Structure):
        _fields_ = [
            ("GetApi",           ctypes.CFUNCTYPE(VP, ctypes.c_uint32)),
            ("GetVersionString", ctypes.CFUNCTYPE(ctypes.c_char_p)),
        ]
    _ort_so.OrtGetApiBase.restype  = ctypes.POINTER(OrtApiBase)
    _ort_so.OrtGetApiBase.argtypes = []
    base = _ort_so.OrtGetApiBase()
    print(f"ORT version: {base.contents.GetVersionString().decode()}")
    _G = base.contents.GetApi(18)


def _add_config(opts, key: str, value: str):
    """AddSessionConfigEntry wrapper。"""
    fn = ctypes.cast(_get_fn(I_AddSessionConfigEntry),
                     ctypes.CFUNCTYPE(VP, VP, ctypes.c_char_p, ctypes.c_char_p))
    _chk(fn(opts, key.encode(), value.encode()))


# opt_level: 2
# global inter op threads: 23, global intra op threads: 1

def make_session(model_path: str, opt_level: int = 2,
                 gpu_stream_count: int = 1, model_directory: str = ""):
    # 1) Env — 使用全局线程池（DisablePerSessionThreads 需要）
    CreateThreadingOpts = ctypes.cast(_get_fn(I_CreateThreadingOptions),
                                      ctypes.CFUNCTYPE(VP, ctypes.POINTER(VP)))
    tp_opts = VP()
    _chk(CreateThreadingOpts(ctypes.byref(tp_opts)))

    SetGlobalIntra = ctypes.cast(_get_fn(I_SetGlobalIntraOpNumThreads),
                                 ctypes.CFUNCTYPE(VP, VP, ctypes.c_int))
    _chk(SetGlobalIntra(tp_opts, 1))

    SetGlobalInter = ctypes.cast(_get_fn(I_SetGlobalInterOpNumThreads),
                                 ctypes.CFUNCTYPE(VP, VP, ctypes.c_int))
    _chk(SetGlobalInter(tp_opts, 23))

    CreateEnv = ctypes.cast(_get_fn(I_CreateEnvWithGlobalThreadPools),
                            ctypes.CFUNCTYPE(VP, ctypes.c_int, ctypes.c_char_p,
                                             VP, ctypes.POINTER(VP)))
    env = VP()
    _chk(CreateEnv(3, b"infer_test", tp_opts, ctypes.byref(env)))

    ReleaseThreadingOpts = ctypes.cast(_get_fn(I_ReleaseThreadingOptions),
                                       ctypes.CFUNCTYPE(None, VP))
    ReleaseThreadingOpts(tp_opts)

    # 2) SessionOptions
    CreateOpts = ctypes.cast(_get_fn(I_CreateSessOpts),
                             ctypes.CFUNCTYPE(VP, ctypes.POINTER(VP)))
    opts = VP()
    _chk(CreateOpts(ctypes.byref(opts)))

    # 3) Graph optimization level
    SetOpt = ctypes.cast(_get_fn(I_SetGraphOptLevel),
                         ctypes.CFUNCTYPE(VP, VP, ctypes.c_int))
    _chk(SetOpt(opts, opt_level))

    # 4) Threading: intra=1, inter=23, parallel mode
    SetIntra = ctypes.cast(_get_fn(I_SetIntraOpNumThreads),
                           ctypes.CFUNCTYPE(VP, VP, ctypes.c_int))
    _chk(SetIntra(opts, 1))

    SetInter = ctypes.cast(_get_fn(I_SetInterOpNumThreads),
                           ctypes.CFUNCTYPE(VP, VP, ctypes.c_int))
    _chk(SetInter(opts, 23))

    # ORT_PARALLEL = 1
    SetMode = ctypes.cast(_get_fn(I_SetExecMode),
                          ctypes.CFUNCTYPE(VP, VP, ctypes.c_int))
    _chk(SetMode(opts, 1))

    # 5) DisablePerSessionThreads — 使用全局线程池
    DisablePST = ctypes.cast(_get_fn(I_DisablePerSessionThreads),
                             ctypes.CFUNCTYPE(VP, VP))
    _chk(DisablePST(opts))

    # 6) DisableCpuMemArena
    DisableArena = ctypes.cast(_get_fn(I_DisableCpuMemArena),
                               ctypes.CFUNCTYPE(VP, VP))
    _chk(DisableArena(opts))

    # 7) Session config entries (对齐 ort_model_test 默认值)
    _add_config(opts, "session.ortvalue_cache_enable", "0")
    _add_config(opts, "session.ortvalue_cache_size", "32")
    _add_config(opts, "session.gpu_use_independent_stream_allocator", "1")
    _add_config(opts, "session.gpu_independent_stream_allocator_max_alloc_ratio", "0.8")
    _add_config(opts, "session.graph_partition_mode", "6")
    _add_config(opts, "session.gpu_parallel_run_mode", "0")
    _add_config(opts, "session.graph_partition_gpu_stream_count", str(gpu_stream_count))
    _add_config(opts, "session.subgraph_reallocation_batch_memcpy_h2d_on", "1")
    _add_config(opts, "session.subgraph_reallocation_batch_memcpy_d2h_on", "1")
    _add_config(opts, "session.memcpy_use_memory_pool_on", "1")
    if model_directory:
        _add_config(opts, "session.model_directory", model_directory)

    # 8) 注册自研自定义算子
    custom = ctypes.CDLL(f"{ORT_LIB}/libcustom_op_lib.so", mode=ctypes.RTLD_GLOBAL)
    custom.RegisterCustomOps.restype  = VP
    custom.RegisterCustomOps.argtypes = [VP, VP]
    _chk(custom.RegisterCustomOps(opts, _ort_so.OrtGetApiBase()))

    # 9) CUDA EP V2 — 支持 tf32/fp16/cuda_graph 等选项
    CreateCudaOpts = ctypes.cast(_get_fn(I_CreateCUDAProviderOptions),
                                 ctypes.CFUNCTYPE(VP, ctypes.POINTER(VP)))
    cuda_opts = VP()
    _chk(CreateCudaOpts(ctypes.byref(cuda_opts)))

    # UpdateCUDAProviderOptions(cuda_opts, keys, values, num_keys)
    UpdateCudaOpts = ctypes.cast(_get_fn(I_UpdateCUDAProviderOptions),
                                 ctypes.CFUNCTYPE(VP, VP,
                                                  ctypes.POINTER(ctypes.c_char_p),
                                                  ctypes.POINTER(ctypes.c_char_p),
                                                  ctypes.c_size_t))
    cuda_keys = (ctypes.c_char_p * 5)(b"device_id", b"enable_cuda_graph",
                                       b"use_tf32", b"use_fp16",
                                       b"provider_blacklist")
    cuda_vals = (ctypes.c_char_p * 5)(b"0", b"0", b"1", b"1", b"black_list_v2.txt")
    _chk(UpdateCudaOpts(cuda_opts, cuda_keys, cuda_vals, 5))

    AppendCudaV2 = ctypes.cast(_get_fn(I_AppendExecutionProvider_CUDA_V2),
                               ctypes.CFUNCTYPE(VP, VP, VP))
    _chk(AppendCudaV2(opts, cuda_opts))

    ReleaseCudaOpts = ctypes.cast(_get_fn(I_ReleaseCUDAProviderOptions),
                                  ctypes.CFUNCTYPE(None, VP))
    ReleaseCudaOpts(cuda_opts)

    # 10) Session
    CreateSess = ctypes.cast(_get_fn(I_CreateSession),
                             ctypes.CFUNCTYPE(VP, VP, ctypes.c_char_p, VP,
                                              ctypes.POINTER(VP)))
    sess = VP()
    _chk(CreateSess(env, model_path.encode(), opts, ctypes.byref(sess)))
    return env, opts, sess


def _get_input_output_names(sess, count_idx, name_idx):
    GetCount = ctypes.cast(_get_fn(count_idx),
                           ctypes.CFUNCTYPE(VP, VP, ctypes.POINTER(ctypes.c_size_t)))
    GetName = ctypes.cast(_get_fn(name_idx),
                          ctypes.CFUNCTYPE(VP, VP, ctypes.c_size_t, VP,
                                           ctypes.POINTER(ctypes.c_char_p)))
    GetAlloc = ctypes.cast(_get_fn(I_GetAllocatorWithDefault),
                           ctypes.CFUNCTYPE(VP, ctypes.POINTER(VP)))
    AllocFree = ctypes.cast(_get_fn(I_AllocatorFree),
                            ctypes.CFUNCTYPE(VP, VP, ctypes.c_void_p))

    n = ctypes.c_size_t()
    _chk(GetCount(sess, ctypes.byref(n)))

    alloc = VP()
    _chk(GetAlloc(ctypes.byref(alloc)))

    names = []
    for i in range(n.value):
        c_name = ctypes.c_char_p()
        _chk(GetName(sess, i, alloc, ctypes.byref(c_name)))
        names.append(c_name.value.decode())
        _chk(AllocFree(alloc, c_name))
    return names


def get_input_names(sess):
    return _get_input_output_names(sess, I_SessionGetInputCount, I_SessionGetInputName)


def get_output_names(sess):
    return _get_input_output_names(sess, I_SessionGetOutputCount, I_SessionGetOutputName)


def create_cpu_memory_info():
    CreateMI = ctypes.cast(_get_fn(I_CreateCpuMemoryInfo),
                           ctypes.CFUNCTYPE(VP, ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(VP)))
    mem_info = VP()
    # OrtArenaAllocator=0, OrtMemTypeDefault=0
    _chk(CreateMI(0, 0, ctypes.byref(mem_info)))
    return mem_info


def numpy_to_ort_value(arr: np.ndarray, mem_info):
    """将 numpy array 转为 OrtValue（CPU tensor）。"""
    # 用 kind 判断 dtype，避免平台差异（np.int64 可能是 longlong）
    kind_to_onnx = {'f': 1, 'i': 6, 'u': 7}  # float→1, int→6(int32), uint/long→7(int64)
    kind = arr.dtype.kind
    onnx_dtype = kind_to_onnx.get(kind, 7)  # fallback int64

    # 精确匹配：float32→1, int16→5, int32→6, int64→7
    if arr.dtype == np.float32:
        onnx_dtype = 1
    elif arr.dtype == np.int16:
        onnx_dtype = 5
    elif arr.dtype == np.int32:
        onnx_dtype = 6
    elif arr.dtype == np.int64:
        onnx_dtype = 7
    elif kind == 'f' and arr.dtype.itemsize == 4:
        onnx_dtype = 1
    elif kind == 'i' and arr.dtype.itemsize == 2:
        onnx_dtype = 5
        arr = arr.astype(np.int16)
    elif kind == 'i' and arr.dtype.itemsize == 4:
        onnx_dtype = 6
    else:
        arr = arr.astype(np.int64)
        onnx_dtype = 7

    arr = np.ascontiguousarray(arr)
    shape = arr.shape
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)

    CreateTensor = ctypes.cast(_get_fn(I_CreateTensorWithData),
                               ctypes.CFUNCTYPE(VP, VP,
                                                ctypes.c_void_p, ctypes.c_size_t,
                                                ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
                                                ctypes.c_int,
                                                ctypes.POINTER(VP)))
    ort_val = VP()
    data_ptr = arr.ctypes.data
    data_len = arr.nbytes
    _chk(CreateTensor(mem_info, data_ptr, data_len,
                      shape_arr, len(shape), onnx_dtype, ctypes.byref(ort_val)))
    return ort_val


def ort_value_to_numpy(ort_val):
    """将 OrtValue 转回 numpy array。"""
    GetTTS = ctypes.cast(_get_fn(I_GetTensorTypeAndShape),
                         ctypes.CFUNCTYPE(VP, VP, ctypes.POINTER(VP)))
    ReleaseTTS = ctypes.cast(_get_fn(I_ReleaseTensorTypeAndShape),
                             ctypes.CFUNCTYPE(None, VP))

    tts = VP()
    _chk(GetTTS(ort_val, ctypes.byref(tts)))

    GetElemType = ctypes.cast(_get_fn(I_GetTensorElementType),
                              ctypes.CFUNCTYPE(VP, VP, ctypes.POINTER(ctypes.c_int32)))
    elem_type = ctypes.c_int32()
    _chk(GetElemType(tts, ctypes.byref(elem_type)))

    ONNX_TO_NP = {1: np.float32, 6: np.int32, 7: np.int64}
    np_dtype = ONNX_TO_NP.get(elem_type.value, np.float32)

    GetData = ctypes.cast(_get_fn(I_GetTensorMutableData),
                          ctypes.CFUNCTYPE(VP, VP, ctypes.POINTER(ctypes.c_void_p)))
    raw_ptr = ctypes.c_void_p()
    _chk(GetData(ort_val, ctypes.byref(raw_ptr)))

    ReleaseTTS(tts)

    # 需要知道 shape 来构建 numpy array — 用 GetDimensions
    GetDimCount = ctypes.cast(_get_fn(I_GetDimensionsCount),
                              ctypes.CFUNCTYPE(VP, VP, ctypes.POINTER(ctypes.c_size_t)))
    GetDims = ctypes.cast(_get_fn(I_GetDimensions),
                          ctypes.CFUNCTYPE(VP, VP, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t))

    tts2 = VP()
    _chk(GetTTS(ort_val, ctypes.byref(tts2)))
    ndim = ctypes.c_size_t()
    _chk(GetDimCount(tts2, ctypes.byref(ndim)))
    dims = (ctypes.c_int64 * ndim.value)()
    _chk(GetDims(tts2, dims, ndim.value))
    ReleaseTTS(tts2)

    shape = [dims[i] for i in range(ndim.value)]
    total = 1
    for d in shape:
        total *= d
    if total == 0:
        return np.array([], dtype=np_dtype)

    buf = (ctypes.c_byte * (total * np.dtype(np_dtype).itemsize)).from_address(raw_ptr.value)
    return np.frombuffer(buf, dtype=np_dtype).reshape(shape).copy()


def get_model_input_dtypes(onnx_path):
    """用 onnx Python 库读取模型输入的 elem_type，返回 {name: elem_type}。"""
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    result = {}
    for inp in m.graph.input:
        et = inp.type.tensor_type.elem_type
        result[inp.name] = et
    del m
    return result


# ONNX elem type → numpy dtype
ONNX_TO_NP_DTYPE = {1: np.float32, 5: np.int16, 6: np.int32, 7: np.int64}


class ORTSession:
    """封装 ORT session 创建与推理，run() 只计时 OrtApi::Run 调用本身。"""

    def __init__(self, model_path: str, opt_level: int = 2,
                 gpu_stream_count: int = 1):
        init_ort()
        self._onnx_path = model_path
        model_dir = os.path.dirname(model_path)
        self.env, self.opts, self.sess = make_session(
            model_path, opt_level=opt_level,
            gpu_stream_count=gpu_stream_count,
            model_directory=model_dir,
        )
        self.input_names = get_input_names(self.sess)
        self.output_names = get_output_names(self.sess)
        self._mem_info = create_cpu_memory_info()

        # 预缓存 Run 函数指针
        self._RunFn = ctypes.cast(_get_fn(I_Run),
                                  ctypes.CFUNCTYPE(VP, VP, VP,
                                                   ctypes.POINTER(ctypes.c_char_p),
                                                   ctypes.POINTER(VP), ctypes.c_size_t,
                                                   ctypes.POINTER(ctypes.c_char_p),
                                                   ctypes.c_size_t,
                                                   ctypes.POINTER(VP)))
        self._ReleaseVal = ctypes.cast(_get_fn(I_ReleaseValue),
                                       ctypes.CFUNCTYPE(None, VP))

    def run(self, feed_dict: dict):
        """执行推理。返回 (output_vals, run_time_s)。

        feed_dict: {input_name: numpy_array}
        output_vals: ctypes 数组，含 output OrtValue 指针
        run_time_s: 仅 OrtApi::Run 调用的耗时（秒）
        """
        input_names = list(feed_dict.keys())
        n_inputs = len(input_names)

        # 准备输入 OrtValue（不计入 run time）
        c_names = (ctypes.c_char_p * n_inputs)(
            *[name.encode() for name in input_names]
        )
        ort_inputs = []
        input_vals = (VP * n_inputs)()
        for i, name in enumerate(input_names):
            ov = numpy_to_ort_value(feed_dict[name], self._mem_info)
            ort_inputs.append(ov)
            input_vals[i] = ov

        n_out = len(self.output_names)
        c_out_names = (ctypes.c_char_p * n_out)(
            *[name.encode() for name in self.output_names]
        )
        output_vals = (VP * n_out)()

        # 仅计时 OrtApi::Run
        t0 = time.perf_counter()
        _chk(self._RunFn(self.sess, VP(), c_names, input_vals, n_inputs,
                          c_out_names, n_out, output_vals))
        run_time = time.perf_counter() - t0

        # 释放 input OrtValues
        for ov in ort_inputs:
            self._ReleaseVal(ov)

        return output_vals, run_time


def main():
    # 加载模型
    print(f"Loading model: {ONNX_PATH}")
    ort_sess = ORTSession(ONNX_PATH, opt_level=2)
    print(f"Session created OK. handle=0x{ort_sess.sess.value:x}")
    print(f"Model inputs: {len(ort_sess.input_names)}, outputs: {len(ort_sess.output_names)}")

    # 选一条 npz 请求
    npz_files = sorted(f for f in os.listdir(REQ_DIR) if f.endswith(".npz"))
    if not npz_files:
        print(f"No npz files found in {REQ_DIR}")
        return
    npz_path = os.path.join(REQ_DIR, npz_files[0])
    print(f"Loading request: {npz_path}")
    data = np.load(npz_path)
    print(f"  npz keys: {len(list(data.keys()))}")

    # 构建 feed_dict
    input_types = get_model_input_dtypes(ONNX_PATH)
    feed_dict = {}
    for name in ort_sess.input_names:
        if name in data:
            arr = data[name]
            expected = ONNX_TO_NP_DTYPE.get(input_types.get(name, 0))
            if expected and arr.dtype != expected:
                arr = arr.astype(expected)
            feed_dict[name] = arr
        else:
            print(f"  WARNING: input '{name}' not in npz, skipping")
    print(f"  matched {len(feed_dict)}/{len(ort_sess.input_names)} inputs")

    # 推理
    print("Running inference...")
    output_vals, run_time = ort_sess.run(feed_dict)
    print(f"ORT Run done in {run_time:.3f}s")

    # 解码输出
    print("\n=== Outputs ===")
    for i in range(len(output_vals)):
        try:
            arr = ort_value_to_numpy(output_vals[i])
            oname = ort_sess.output_names[i] if i < len(ort_sess.output_names) else f"out_{i}"
            print(f"  {oname}: shape={list(arr.shape)} dtype={arr.dtype} "
                  f"min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")
        except Exception as e:
            print(f"  output[{i}]: decode failed: {e}")
        finally:
            ort_sess._ReleaseVal(output_vals[i])


if __name__ == "__main__":
    main()
