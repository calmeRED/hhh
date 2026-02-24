import pathlib
import pdb
import random
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
import matplotlib.pyplot as plt
from collections import defaultdict

from timm.utils import init_distributed_device
from tqdm import trange

# ----------------------------------------------------------------------
# 随机采样工具
# ----------------------------------------------------------------------
def sample_random_values(
    spec: Dict[str, List[Any]],
    *,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    根据 ``spec`` 随机生成变量取值。

    ``spec`` 的每一项必须是 ``[low, high, dtype]``，其中
    ``dtype`` 支持 ``Real``, ``Integer``, ``Boolean`` 与 ``Radians``。

    Parameters
    ----------
    spec : dict
        变量规格字典，键为变量名，值为 ``[low, high, dtype]``。
    seed : int, optional
        随机种子，若提供则同时为 ``numpy`` 与 ``random`` 设置。

    Returns
    -------
    dict
        变量名 → 随机数值 的映射。
    """
    def _rand_real(low: float, high: float) -> float:
        return float(np.random.uniform(low, high))

    def _rand_int(low: int, high: int) -> int:
        # ``np.random.randint`` 的上界是开区间，需要 +1
        return int(np.random.randint(low, high + 1))

    def _rand_bool() -> bool:
        return bool(random.getrandbits(1))

    def _rand_radians(low, high):
        low_arr = np.asarray(low, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        if np.any(low_arr > high_arr):
            raise ValueError("在 radians 采样中出现 low > high 的情况")
        return np.random.uniform(low_arr, high_arr)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    sample: Dict[str, Any] = {}
    for var_name, cfg in spec.items():
        if not isinstance(cfg, (list, tuple)) or len(cfg) != 3:
            print(f"变量 '{var_name}' 的规格必须是长度为 3 的 list/tuple，得到 {cfg}")
            continue

        low, high, dtype = cfg
        if dtype is None:
            print(f"变量 '{var_name}' 缺少 dtype（'Real'/'Integer'/'Boolean'）")
            continue

        dtype = str(dtype).strip().lower()
        value: Any = None

        if dtype == "real":
            low_f, high_f = float(low), float(high)
            if low_f > high_f:
                raise ValueError("low > high")
            value = _rand_real(low_f, high_f)

        elif dtype == "integer":
            low_i, high_i = int(low), int(high)
            if low_i > high_i:
                raise ValueError("low > high")
            value = _rand_int(low_i, high_i)

        elif dtype == "boolean":
            value = _rand_bool()

        elif dtype == "radians":
            value = _rand_radians(low, high)

        else:
            print(f"变量 '{var_name}' 的 dtype '{dtype}' 不被支持，仅支持 Real/Integer/Boolean/Radians")
            continue

        sample[var_name] = value

    return sample


# ----------------------------------------------------------------------
# FMU 环境
# ----------------------------------------------------------------------
class FMUITMS:
    """
    通用的 FMU 环境（Co‑Simulation）
    * 自动解压、读取 ModelDescription、实例化 FMU；
    * 根据 ``observation_list`` 只读取感兴趣的输出；
    * 默认的 reward_fn / terminate_fn；
    * 支持 ``init_dict``（可在 reset 时写入），以及随机种子；
    * 完全兼容 ``gymnasium.Env`` 接口（observation_space / action_space 等）。
    """
    # ------------------------------------------------------------------
    def __init__(
        self,
        fmu_path: str | pathlib.Path,
        step_size: float,
        target_state: Optional[np.ndarray] = None,
        max_episode_steps: Optional[int] = 1800,
        init_dict: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        observation_list: Optional[List[str]] = None,
        action_bounds: Dict = None,
    ):
        # ------------------- 基础属性 -------------------
        self.fmu_path = pathlib.Path(fmu_path).absolute()
        self.step_size = float(step_size)
        self.max_episode_steps = max_episode_steps
        self.observation_list = observation_list
        self.action_bounds = action_bounds
        self.init_dict = init_dict
        # ------------------- 随机种子 -------------------
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # ------------------- 解压 & 读取 ModelDescription -------------------
        self.unzip_dir = extract(self.fmu_path)  # 解压到临时目录
        self.md = read_model_description(self.fmu_path, validate=False)

        # ------------------- 检查 Co‑Simulation 接口 -------------------
        self.cs = self.md.coSimulation
        if self.cs is None:
            raise RuntimeError(
                "该 FMU 只实现了 ModelExchange，FMUITMS 只能处理 CoSimulation FMU"
            )

        # ------------------- 变量映射 -------------------
        # name → (valueReference, fmi_type)
        self.vrs: Dict[str, Tuple[int, str]] = {}
        self.input_names: List[str] = []   # 可写变量（input / parameter）
        self.output_names: List[str] = []  # 读取变量（output / local / calculatedParameter）

        for var in self.md.modelVariables:
            self.vrs[var.name] = (var.valueReference, var.type)

            if var.causality in ("input", "parameter"):
                self.input_names.append(var.name)
            elif var.causality in ("output", "local", "calculatedParameter"):
                self.output_names.append(var.name)

        # ------------------- 目标状态 -------------------
        if target_state is None:
            self.target_state = np.zeros(len(self.output_names), dtype=np.float64)
        else:
            self.target_state = np.asarray(target_state, dtype=np.float64)

        if self.target_state.shape[0] != len(self.output_names):
            raise ValueError(
                f"target_state 长度 {self.target_state.shape[0]} 与 FMU 输出维度 "
                f"{len(self.output_names)} 不匹配"
            )



        # ------------------- 内部状态 -------------------
        self._last_obs: Optional[Dict[str, Any]] = None
        self.current_step: int = 0
        self.current_time: float = 0.0

        # ------------------- 创建 FMU 实例 -------------------
        self._create_fmu_instance()

        # ------------------- 默认 reward / terminate -------------------
        self.terminate_fn = self._default_terminate


    # ------------------------------------------------------------------
    # FMU 实例化（内部复用）
    # ------------------------------------------------------------------
    def _create_fmu_instance(self) -> None:
        """
        创建（或重新创建）FMU 实例并完成基本的 instantiate / init 步骤。
        """
        self._fmu = FMU2Slave(
            guid=self.md.guid,
            unzipDirectory=str(self.unzip_dir),
            modelIdentifier=self.cs.modelIdentifier,
            instanceName="fmu_itms_instance",
        )
        self._fmu.instantiate()
        self._fmu.setupExperiment(startTime=0.0)
        self._fmu.enterInitializationMode()
        if self.init_dict:
            for k, v in self.init_dict.items():
                self._fmu.setReal([self.vrs[k][0]], [float(v)])
        self._fmu.exitInitializationMode()

    # ------------------------------------------------------------------
    # 读取输出
    # ------------------------------------------------------------------
    def read_outputs(self, param_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        param_list : list of str, optional
            只读取指定变量；若为 ``None`` 则读取全部 ``self.output_names``。
        """
        values: Dict[str, Any] = {}
        names = param_list if param_list else self.output_names
        for name in names:
            vr, typ = self.vrs[name]
            if typ in ("Real", "Radians"):
                values[name] = self._fmu.getReal([vr])[0]
            elif typ == "Integer":
                values[name] = self._fmu.getInteger([vr])[0]
            elif typ == "Boolean":
                # 把布尔值转成 Python bool，兼容 gym 的 observation
                values[name] = bool(self._fmu.getBoolean([vr])[0])
            elif typ == "String":
                # FMU 中的字符串在大多数控制任务里不需要，直接跳过
                continue
            else:
                raise TypeError(f"Unsupported output type {typ} for variable {name}")
        return values

    # ------------------------------------------------------------------
    # 写入动作 / 参数
    # ------------------------------------------------------------------
    def set_action(self, action_dict: Dict[str, Any], check_bounds: bool = True) -> None:
        """
        action_dict : dict
            变量名 → 待写入的数值。
        check_bounds : bool, optional
            若为 ``True``，会根据 ModelVariable 中的 ``min`` / ``max`` 检查范围。
        """
        for var_name, value in action_dict.items():
            if var_name not in self.input_names:
                print(f"变量 '{var_name}' 不是 FMU 的输入（可写变量列表参阅self.input_names)")
                continue

            vr, fmi_type = self.vrs[var_name]

            if check_bounds:
                # 读取 ModelVariable（只在第一次需要时缓存）
                if not hasattr(self, "_md_var_cache"):
                    self._md_var_cache = {v.name: v for v in self.md.modelVariables}
                md_var = self._md_var_cache[var_name]

                if md_var.type in ("Real", "Integer"):
                    low = md_var.min if md_var.min is not None else -np.inf
                    high = md_var.max if md_var.max is not None else np.inf
                    if not (low <= value <= high):
                        raise ValueError(
                            f"变量 {var_name} 的值 {value} 超出声明范围 [{low}, {high}]"
                        )
                if md_var.type == "Boolean" and value not in (0, 1, False, True):
                    raise ValueError(f"Boolean 变量 {var_name} 只能取 0/1 或 False/True")

            # 实际写入 FMU
            try:
                if fmi_type == "Real":
                    self._fmu.setReal([vr], [float(value)])
                elif fmi_type == "Integer":
                    self._fmu.setInteger([vr], [int(value)])
                elif fmi_type == "Boolean":
                    self._fmu.setBoolean([vr], [bool(value)])
                else:
                    raise TypeError(f"Unsupported FMU input type '{fmi_type}' for '{var_name}'")
            except Exception as exc:
                raise RuntimeError(
                    f"向 FMU 写入变量 '{var_name}' (vr={vr}, type={fmi_type}) 时出错: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # 默认 reward / terminate（基于目标状态）
    # ------------------------------------------------------------------
    def _default_terminate(self, obs: Dict[str, Any]) -> bool:
        """这个仿真环境下没有终止条件，除非FMU运行报错了得重开"""
        # if obs["battery.controlBus.batteryBus.battery_SOC[1]"] < 1:
        #     return True
        return False

    # ------------------------------------------------------------------
    # 环境核心 API：reset / step / render / close
    # ------------------------------------------------------------------

    def reset(self, init_dict, random_perturb = True):
        """
        重置环境（重新实例化 FMU），并把 ``init_dict`` 写入。

        options : dict, optional
            若提供 ``options['random_init'] == True``，则在每次 reset 时把
            ``init_dict`` 中的数值在 ±10% 范围内随机扰动（仅示例）。

        """
        self.init_dict = init_dict
        # 重置计数器
        self.current_step = 0
        self.current_time = 0.0
        # 释放旧实例（如果已经创建过）
        try:
            self._fmu.terminate()
        except Exception:
            pass
        #  重新创建实例并完成初始化模式
        self._create_fmu_instance()
        # 读取首次观测
        obs = self.read_outputs(self.observation_list)
        self._last_obs = obs
        return obs

    def step(self, action_dict: Dict[str, Any]):
        """
        执行一次仿真步。
        - 先把 ``action_dict`` 写入 FMU（如果有）。
        - 调用 ``doStep`` 前进 ``step_size``。
        - 读取输出、计算 reward、判断终止/截断。
        - 若 episode 结束，自动调用 ``reset``。
        """

        terminated = False
        truncated = False
        new_obs = None

        # 写入动作（如果提供）
        if action_dict:
            self.set_action(action_dict, check_bounds=True)

        try:
            # 前进一步，可能要出错
            self._fmu.doStep(self.current_time, self.step_size)
            # 读取新状态
            new_obs = self.read_outputs(self.observation_list)

            # 计算 reward / terminate / truncated

            # terminated = (
            #     self.terminate_fn(new_obs) if self.terminate_fn else False
            # )
            truncated = (
                    self.max_episode_steps is not None
                    and self.current_step + 1 >= self.max_episode_steps
            )
            # 更新时间
            self.current_step += 1
            self.current_time += self.step_size
            self._last_obs = new_obs

        except Exception as e:
            terminated = True
            print(f"My Error Message: {e}")

        # 自动 reset（符合 gymnasium 的 “auto‑reset” 约定）
        if terminated or truncated:
            new_obs = self.reset()

        return new_obs, bool(terminated), bool(truncated)



    def render(self, mode: str = "human"):
        pass

    def close(self):
        """释放 FMU 资源。"""
        try:
            self._fmu.terminate()
        except Exception:
            pass


#================================================================
#================================================================
def K_to_C(values):
    return [v - 273.15 for v in values]
def C_to_K(values):
    return [v + 273.15 for v in values]
def Pa_to_kPa(values):
    return [v / 1_000 for v in values]
UNIT_MAP = {               # 原单位 → (转换函数, 绘图时的单位标签)
    "K" : (K_to_C , "°C"),
    "Pa": (Pa_to_kPa, "kPa"),
    None: (lambda v: v, None),
}

def run_and_plot_multi(env, cfg, steps=1500, step_time=None):
    data = {fig:{var:{"unit":u,"vals":[]} for var,(u,_) in vars.items()}
            for fig,vars in cfg.items()}

    t = []
    for s in trange(steps):
        obs, term, trunc = env.step({})
        t.append(s if step_time is None else s*step_time)

        for fig in data:
            for var in data[fig]:
                data[fig][var]["vals"].append(obs[var])

        if term or trunc: break

    # ---------- 绘图 ----------
    for fig_name, vars in data.items():
        plt.figure(fig_name)               # 新建/切换 Figure
        ax_l = plt.gca()
        ax_r = None
        first = True

        # 按单位分组，第一种单位用左 y 轴，第二种（若有）用右 y 轴
        groups = defaultdict(list)
        for v,meta in vars.items():
            groups[meta["unit"]].append((v,meta["vals"]))

        for unit, var_list in groups.items():
            ax = ax_l if first else (ax_r or ax_l.twinx())
            if not first: ax_r = ax
            conv, label_u = UNIT_MAP.get(unit, (lambda x:x, unit))

            for name, raw in var_list:
                y = conv(raw)
                lab = f"{name} ({label_u})" if label_u else name
                ax.plot(t, y, label=lab)

            first = False

        # 合并左右轴图例
        handles, labs = ax_l.get_legend_handles_labels()
        if ax_r:
            h2, l2 = ax_r.get_legend_handles_labels()
            handles, labs = handles+h2, labs+l2
        ax_l.legend(handles, labs, loc="best")
        plt.xlabel("step")
        plt.title(fig_name)
        plt.tight_layout()
        plt.show()

# ---------- 使用示例 ----------
if __name__ == "__main__":
    cfg = {
        "Temp.": {
        #     "T_Amb":               ["K", None],
            "battery.Batt_top[1].T": ["K", None],
            "cabinVolume.summary.T": ["K", None],
        },
        "W.C.": {
            "driverPerformance.controlBus.driverBus._acc_pedal_travel": ["%", None],
            "driverPerformance.controlBus.driverBus._brake_pedal_travel": ["%", None],
            "battery.controlBus.batteryBus.battery_SOC[1]": ["%", None],
            # "driverPerformance.controlBus.vehicleStatus.vehicle_velocity": ["m/s", None],
        },
    }

    fmu_file_path = "MyITMS.fmu"
    action_bounds = {
        'BattPump_rpm': [0, 3000],
        'MotorPump_rpm': [0,3000],
        'ComprSpd_rpm': [0, 10000],
        'valve_pos1': [True, False],
        'valve_pos2': [True, False],
    }
    observation_list = [v for fig in cfg.values() for v in fig]
    print(observation_list)
    env = FMUITMS(
        fmu_path=fmu_file_path,
        step_size=1,
        observation_list=observation_list,
        action_bounds=None,
        seed=42,
        init_dict=None,
    )

    init_dict = {"MY_socinit": 0.123,
                 "MY_battT0": 0.123,
                 "MY_motorT0": 0.123,
                 # "T_Cabin": 0.123,
                 # "T_out": 0.123,
                }
    env.reset(init_dict, False)
    env.set_action({'RPM_batt': 0.123})
    run_and_plot_multi(env, cfg, steps=20) #, step_time=0.02)

    env.close()