"""
循环优化求解器 - 遗传算法
- 实数编码
- 约束处理: 涡轮出口干度>=0.88, 状态点温度不超过工质上限
- 优化目标: 最大化热效率 或 最大化净功
"""

import numpy as np
from copy import deepcopy

from thermo.steam import T_MAX_STEAM
from thermo.ideal_gas import GAS_PARAMS


MIN_QUALITY = 0.88


def _build_cycle_from_params(cfg_key, param_dict, CYCLE_CONFIGS):
    """根据配置键和参数构建循环对象"""
    cfg = CYCLE_CONFIGS[cfg_key]
    cls = cfg['class']
    
    kwargs = {}
    for p in cfg['params']:
        val = param_dict.get(p['key'], p['default'])
        if p['key'].startswith('T'):
            kwargs[p['key']] = val + 273.15
        else:
            kwargs[p['key']] = val
    
    if 'extra_kwargs' in cfg:
        kwargs.update(cfg['extra_kwargs'])
    
    cycle = cls(**kwargs)
    results = cycle.compute()
    return cycle, results


def _check_constraints(cfg_key, cycle, results, CYCLE_CONFIGS):
    """
    检查约束条件:
    1. 涡轮出口干度 >= 0.88 (对水蒸气循环)
    2. 所有状态点温度 <= 工质上限
    返回: (满足约束?, 违规信息列表)
    """
    violations = []
    
    cfg = CYCLE_CONFIGS[cfg_key]
    fluid_type = cfg.get('fluid_type', _detect_fluid(cfg_key))
    
    # ---- 约束1: 涡轮出口干度 (只对水蒸气循环) ----
    if fluid_type in ('water', 'mixed'):
        x_out = results.get('x_turbine_out')
        if x_out is not None and x_out < MIN_QUALITY:
            violations.append(f"涡轮出口干度={x_out:.4f} < {MIN_QUALITY}")
    
    # ---- 约束2: 状态点温度不超过工质上限 ----
    T_limits = _get_temp_limits(cfg_key)
    
    max_T_violation = False
    for label, sp in cycle.states.items():
        if sp.T is None:
            continue
        for fluid, T_max in T_limits.items():
            # 根据状态点判断工质类型
            if _state_matches_fluid(sp, fluid, cfg_key):
                if sp.T > T_max:
                    max_T_violation = True
                    violations.append(
                        f"状态点{label}温度={sp.T-273.15:.1f}°C > 上限{T_max-273.15:.0f}°C"
                    )
                    break
    
    # 检查results中的warnings (包含温度超限警告)
    for w in results.get('warnings', []):
        if '超过上限' in w or '过高' in w:
            if w not in violations:
                violations.append(w)
    
    return (len(violations) == 0), violations


def _detect_fluid(cfg_key):
    """根据循环键检测工质类型"""
    water_keys = {'rankine_basic', 'rankine_reheat', 'rankine_regen'}
    gas_keys = {'brayton_basic', 'brayton_regen', 'otto', 'diesel'}
    if cfg_key in water_keys:
        return 'water'
    elif cfg_key in gas_keys:
        return 'gas'
    elif cfg_key == 'ccgt':
        return 'mixed'
    return 'water'


def _get_temp_limits(cfg_key):
    """获取各工质的温度上限"""
    limits = {}
    fluid_type = _detect_fluid(cfg_key)
    
    if fluid_type in ('water', 'mixed'):
        limits['water'] = T_MAX_STEAM
    
    if fluid_type in ('gas', 'mixed'):
        limits['air'] = GAS_PARAMS.get('air', {}).get('T_max', 1773.15)
    
    return limits


def _state_matches_fluid(state_point, fluid, cfg_key):
    """检查状态点是否属于指定工质"""
    if fluid == 'water' and state_point.fluid in ('water', 'steam'):
        return True
    if fluid == 'air' and state_point.fluid in ('air', 'gas', 'argon', 'nitrogen'):
        return True
    return False


def _get_objective_value(results, objective):
    """提取目标值 (eta或w_net)"""
    if objective == 'eta':
        return results.get('eta', results.get('eta_total', 0)) or 0
    else:  # w_net
        return results.get('w_net', results.get('W_dot_total_kW', 0)) or 0


class GeneticOptimizer:
    """遗传算法优化器 - 实数编码"""
    
    def __init__(self, cfg_key, CYCLE_CONFIGS,
                 objective='eta',
                 opt_param_configs=None,
                 fixed_params=None,
                 pop_size=50,
                 n_generations=80,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 progress_callback=None):
        """
        参数:
          cfg_key: 循环配置键 (如 'rankine_basic')
          CYCLE_CONFIGS: 全局循环配置字典
          objective: 'eta' (热效率) 或 'w_net' (净功)
          opt_param_configs: list of dict, 每个含 {key, min, max}
          fixed_params: dict, 固定参数 {key: value}
          pop_size: 种群大小
          n_generations: 进化代数
          crossover_rate: 交叉概率
          mutation_rate: 变异概率
          progress_callback: 回调函数 fn(generation_idx, n_generations, best_fitness)
        """
        self.cfg_key = cfg_key
        self.CYCLE_CONFIGS = CYCLE_CONFIGS
        self.cfg = CYCLE_CONFIGS[cfg_key]
        self.objective = objective
        
        self.opt_params = opt_param_configs or []
        self.fixed_params = fixed_params or {}
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.progress_callback = progress_callback
        
        # 进化记录
        self.generation_best = []
        self.generation_avg = []
        self.best_individual = None
        self.best_fitness = 0
        self.best_results = None
        self.best_cycle = None
        
        # 统计违规情况
        self.valid_count_per_gen = []
    
    def _init_population(self):
        """初始化种群 - 在参数范围内均匀随机采样"""
        n_params = len(self.opt_params)
        population = np.zeros((self.pop_size, n_params))
        
        for i in range(n_params):
            lb = self.opt_params[i]['min']
            ub = self.opt_params[i]['max']
            population[:, i] = np.random.uniform(lb, ub, self.pop_size)
        
        return population
    
    def _decode(self, individual):
        """将个体实数编码解码为参数字典"""
        params = {}
        for i, p_cfg in enumerate(self.opt_params):
            key = p_cfg['key']
            params[key] = individual[i]
        params.update(self.fixed_params)
        return params
    
    def _evaluate(self, individual):
        """
        评估单个个体
        返回: fitness (违约个体为0), 计算结果或None, 是否满足约束
        """
        param_dict = self._decode(individual)
        
        try:
            cycle, results = _build_cycle_from_params(
                self.cfg_key, param_dict, self.CYCLE_CONFIGS
            )
        except Exception as e:
            return 0.0, None, False, [str(e)]
        
        feasible, violations = _check_constraints(
            self.cfg_key, cycle, results, self.CYCLE_CONFIGS
        )
        
        if not feasible:
            return 0.0, results, False, violations
        
        obj_val = _get_objective_value(results, self.objective)
        
        # 适应度: 目标值必须为正
        fitness = max(obj_val, 1e-10)
        
        return fitness, results, True, []
    
    def _tournament_selection(self, population, fitnesses, tournament_size=3):
        """锦标赛选择"""
        idxs = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = idxs[np.argmax(fitnesses[idxs])]
        return population[best_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """算术交叉 (实数编码)"""
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(len(parent1))
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual):
        """高斯变异 (实数编码)"""
        mutant = individual.copy()
        n_params = len(self.opt_params)
        
        for i in range(n_params):
            if np.random.rand() < self.mutation_rate:
                lb = self.opt_params[i]['min']
                ub = self.opt_params[i]['max']
                # 高斯变异, 步长为范围的10%
                sigma = (ub - lb) * 0.1
                mutant[i] = individual[i] + np.random.normal(0, sigma)
                # 边界裁剪
                mutant[i] = np.clip(mutant[i], lb, ub)
        
        return mutant
    
    def _clip_bounds(self, individual):
        """裁剪到参数边界内"""
        for i in range(len(self.opt_params)):
            lb = self.opt_params[i]['min']
            ub = self.opt_params[i]['max']
            individual[i] = np.clip(individual[i], lb, ub)
        return individual
    
    def run(self):
        """运行遗传算法"""
        if len(self.opt_params) == 0:
            raise ValueError("至少需要指定一个优化参数")
        
        n_params = len(self.opt_params)
        
        # ---- 1. 初始化种群 ----
        population = self._init_population()
        
        # ---- 2. 初始评估 ----
        fitnesses = np.zeros(self.pop_size)
        valid_flags = np.zeros(self.pop_size, dtype=bool)
        results_cache = [None] * self.pop_size
        
        for i in range(self.pop_size):
            fitnesses[i], results_cache[i], valid_flags[i], _ = self._evaluate(population[i])
        
        best_idx = int(np.argmax(fitnesses))
        self.best_individual = population[best_idx].copy()
        self.best_fitness = float(fitnesses[best_idx])
        self.best_results = results_cache[best_idx]
        
        self.generation_best.append(float(self.best_fitness))
        self.generation_avg.append(float(np.mean(fitnesses)))
        self.valid_count_per_gen.append(int(np.sum(valid_flags)))
        
        # 回调 - 初始代数
        if self.progress_callback:
            self.progress_callback(0, self.n_generations, self.best_fitness)
        
        # ---- 3. 进化循环 ----
        for gen in range(1, self.n_generations + 1):
            new_population = np.zeros((self.pop_size, n_params))
            new_fitnesses = np.zeros(self.pop_size)
            new_valid = np.zeros(self.pop_size, dtype=bool)
            new_results = [None] * self.pop_size
            
            # 精英策略: 保留最佳个体
            new_population[0] = self.best_individual.copy()
            new_fitnesses[0] = self.best_fitness
            new_valid[0] = True
            new_results[0] = self.best_results
            
            # 产生剩余个体
            for i in range(1, self.pop_size, 2):
                # 选择
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # 边界裁剪
                child1 = self._clip_bounds(child1)
                child2 = self._clip_bounds(child2)
                
                # 评估
                new_population[i] = child1
                new_fitnesses[i], new_results[i], new_valid[i], _ = self._evaluate(child1)
                
                if i + 1 < self.pop_size:
                    new_population[i + 1] = child2
                    new_fitnesses[i + 1], new_results[i + 1], new_valid[i + 1], _ = self._evaluate(child2)
            
            # 更新种群
            population = new_population
            fitnesses = new_fitnesses
            valid_flags = new_valid
            results_cache = new_results
            
            # 更新最优
            current_best_idx = int(np.argmax(fitnesses))
            if fitnesses[current_best_idx] > self.best_fitness:
                self.best_fitness = float(fitnesses[current_best_idx])
                self.best_individual = population[current_best_idx].copy()
                self.best_results = results_cache[current_best_idx]
            
            self.generation_best.append(float(self.best_fitness))
            self.generation_avg.append(float(np.mean(fitnesses)))
            self.valid_count_per_gen.append(int(np.sum(valid_flags)))
            
            # 进度回调
            if self.progress_callback:
                self.progress_callback(gen, self.n_generations, self.best_fitness)
        
        # ---- 4. 重建最佳个体的循环对象 ----
        best_params = self._decode(self.best_individual)
        try:
            self.best_cycle, self.best_results = _build_cycle_from_params(
                self.cfg_key, best_params, self.CYCLE_CONFIGS
            )
        except:
            pass
        
        return {
            'best_params': best_params,
            'best_fitness': self.best_fitness,
            'best_objective': self.best_fitness,
            'objective_type': self.objective,
            'generation_best': self.generation_best,
            'generation_avg': self.generation_avg,
            'valid_count_per_gen': self.valid_count_per_gen,
            'best_results': self.best_results,
            'n_generations': self.n_generations,
            'pop_size': self.pop_size,
        }
