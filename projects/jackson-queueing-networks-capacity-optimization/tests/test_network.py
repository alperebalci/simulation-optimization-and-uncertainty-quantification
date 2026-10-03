import numpy as np

from jackson_or import CapacityOption, JacksonNetwork, optimize_capacity


def demo_network(service_rates=(3.2, 2.5, 2.0)):
    external = np.array([1.0, 0.2, 0.0])
    routing = np.array([
        [0.0, 0.55, 0.10],
        [0.0, 0.0, 0.45],
        [0.0, 0.0, 0.0],
    ])
    return JacksonNetwork(external, routing, np.asarray(service_rates, dtype=float))


def test_traffic_equations_and_littles_law():
    network = demo_network()
    metrics = network.metrics()
    lhs = metrics.traffic_rates
    rhs = network.external_arrival_rates + network.routing.T @ lhs
    assert np.allclose(lhs, rhs)
    assert np.isclose(
        metrics.total_mean_number,
        metrics.external_throughput * metrics.system_mean_sojourn,
    )


def test_simulation_tracks_product_form_means():
    network = demo_network()
    analytical = network.metrics()
    simulated = network.simulate(horizon=30_000.0, warmup=3_000.0, seed=7)
    assert np.allclose(
        simulated.mean_numbers,
        analytical.mean_numbers,
        rtol=0.25,
        atol=0.08,
    )


def test_capacity_optimization_uses_budget_to_reduce_congestion():
    external = np.array([1.0, 0.2, 0.0])
    routing = np.array([
        [0.0, 0.55, 0.10],
        [0.0, 0.0, 0.45],
        [0.0, 0.0, 0.0],
    ])
    options = [
        [CapacityOption(2.0, 0.0), CapacityOption(3.2, 2.0)],
        [CapacityOption(1.4, 0.0), CapacityOption(2.5, 2.0)],
        [CapacityOption(1.1, 0.0), CapacityOption(2.0, 2.0)],
    ]
    baseline = optimize_capacity(external, routing, options, budget=0.0)
    improved = optimize_capacity(external, routing, options, budget=4.0)
    assert improved.cost <= 4.0
    assert improved.total_mean_number < baseline.total_mean_number
