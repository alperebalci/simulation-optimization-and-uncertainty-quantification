"""Analytical and simulation tools for open M/M/1 Jackson networks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class NetworkMetrics:
    traffic_rates: np.ndarray
    utilizations: np.ndarray
    mean_numbers: np.ndarray
    mean_sojourn_per_visit: np.ndarray
    total_mean_number: float
    external_throughput: float
    system_mean_sojourn: float


@dataclass(frozen=True)
class SimulationResult:
    mean_numbers: np.ndarray
    total_mean_number: float
    elapsed_time: float


@dataclass(frozen=True)
class CapacityOption:
    service_rate: float
    cost: float


@dataclass(frozen=True)
class CapacityPlan:
    option_indices: tuple[int, ...]
    service_rates: np.ndarray
    cost: float
    total_mean_number: float


@dataclass(frozen=True)
class JacksonNetwork:
    external_arrival_rates: np.ndarray
    routing: np.ndarray
    service_rates: np.ndarray

    def __post_init__(self) -> None:
        e = np.asarray(self.external_arrival_rates, dtype=float)
        p = np.asarray(self.routing, dtype=float)
        mu = np.asarray(self.service_rates, dtype=float)
        n = e.size
        if e.shape != (n,) or mu.shape != (n,) or p.shape != (n, n):
            raise ValueError("Inconsistent Jackson-network dimensions.")
        if np.any(e < 0) or np.any(mu <= 0):
            raise ValueError("Arrival rates must be nonnegative and service rates positive.")
        if np.any(p < 0) or np.any(p.sum(axis=1) > 1.0 + 1e-12):
            raise ValueError("Routing rows must be substochastic.")
        object.__setattr__(self, "external_arrival_rates", e)
        object.__setattr__(self, "routing", p)
        object.__setattr__(self, "service_rates", mu)

    def traffic_rates(self) -> np.ndarray:
        """Solve lambda = e + P^T lambda."""
        identity = np.eye(self.routing.shape[0])
        rates = np.linalg.solve(identity - self.routing.T, self.external_arrival_rates)
        if np.any(rates < -1e-10):
            raise RuntimeError("Traffic equations produced a negative rate.")
        return np.maximum(rates, 0.0)

    def metrics(self) -> NetworkMetrics:
        rates = self.traffic_rates()
        rho = rates / self.service_rates
        if np.any(rho >= 1.0):
            raise ValueError("Network is unstable: every node requires lambda_i < mu_i.")
        mean_n = rho / (1.0 - rho)
        mean_visit_time = 1.0 / (self.service_rates - rates)
        total_n = float(mean_n.sum())
        external_rate = float(self.external_arrival_rates.sum())
        system_w = total_n / external_rate if external_rate > 0 else 0.0
        return NetworkMetrics(
            traffic_rates=rates,
            utilizations=rho,
            mean_numbers=mean_n,
            mean_sojourn_per_visit=mean_visit_time,
            total_mean_number=total_n,
            external_throughput=external_rate,
            system_mean_sojourn=system_w,
        )

    def simulate(
        self,
        *,
        horizon: float = 20_000.0,
        warmup: float = 2_000.0,
        seed: int = 2026,
    ) -> SimulationResult:
        """Continuous-time event simulation for a Jackson network."""
        if horizon <= warmup or warmup < 0:
            raise ValueError("Require horizon > warmup >= 0.")
        rng = np.random.default_rng(seed)
        n_nodes = self.external_arrival_rates.size
        state = np.zeros(n_nodes, dtype=int)
        area = np.zeros(n_nodes, dtype=float)
        t = 0.0
        measured_time = 0.0

        while t < horizon:
            service = self.service_rates * (state > 0)
            event_rates = np.concatenate([self.external_arrival_rates, service])
            total_rate = float(event_rates.sum())
            if total_rate <= 0:
                break
            dt = float(rng.exponential(1.0 / total_rate))
            next_t = min(t + dt, horizon)

            left = max(t, warmup)
            right = max(min(next_t, horizon), warmup)
            if right > left:
                duration = right - left
                area += state * duration
                measured_time += duration

            if t + dt >= horizon:
                t = horizon
                break

            draw = float(rng.random() * total_rate)
            event = int(np.searchsorted(np.cumsum(event_rates), draw, side="right"))

            if event < n_nodes:
                state[event] += 1
            else:
                i = event - n_nodes
                state[i] -= 1
                u = float(rng.random())
                cumulative = 0.0
                for j, prob in enumerate(self.routing[i]):
                    cumulative += float(prob)
                    if u < cumulative:
                        state[j] += 1
                        break
            t += dt

        if measured_time <= 0:
            raise RuntimeError("No post-warmup simulation time was accumulated.")
        mean_n = area / measured_time
        return SimulationResult(
            mean_numbers=mean_n,
            total_mean_number=float(mean_n.sum()),
            elapsed_time=measured_time,
        )


def optimize_capacity(
    external_arrival_rates: np.ndarray,
    routing: np.ndarray,
    options_by_node: list[list[CapacityOption]],
    budget: float,
) -> CapacityPlan:
    """Enumerate discrete service-rate investments and minimize steady-state WIP."""
    if budget < 0:
        raise ValueError("budget must be nonnegative.")
    best: CapacityPlan | None = None
    for indices in product(*(range(len(options)) for options in options_by_node)):
        chosen = [options_by_node[i][k] for i, k in enumerate(indices)]
        cost = float(sum(option.cost for option in chosen))
        if cost > budget + 1e-12:
            continue
        rates = np.array([option.service_rate for option in chosen], dtype=float)
        network = JacksonNetwork(external_arrival_rates, routing, rates)
        try:
            total_n = network.metrics().total_mean_number
        except ValueError:
            continue
        candidate = CapacityPlan(indices, rates, cost, total_n)
        if best is None or (candidate.total_mean_number, candidate.cost) < (
            best.total_mean_number,
            best.cost,
        ):
            best = candidate
    if best is None:
        raise ValueError("No stable capacity plan satisfies the budget.")
    return best
