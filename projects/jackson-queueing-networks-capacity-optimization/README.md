# Jackson Queueing Networks and Capacity Optimization

This project adds a queueing-network layer that is distinct from the repository's existing single-station simulation models and Erlang-style staffing examples.

It implements an **open Jackson network** with M/M/1 nodes:

- traffic equations `lambda = e + P^T lambda`;
- node utilizations and product-form mean queue lengths;
- Little's-law system sojourn time;
- continuous-time stochastic event simulation as an independent numerical check;
- discrete capacity-investment optimization over service-rate alternatives.

The optimizer does not replace queueing physics with a generic MILP objective. It first solves the endogenous network traffic rates, rejects unstable capacity plans, and then minimizes analytical steady-state work-in-process subject to an investment budget.

## Why this matters

A set of independent M/M/c calculations misses routed congestion. In a network, downstream arrival rates depend on upstream completions and routing probabilities. The Jackson traffic equations make that interaction explicit.

## Validation

The tests verify:

1. the traffic-flow fixed point;
2. Little's law;
3. simulation agreement with analytical product-form means under a fixed seed;
4. lower congestion from a feasible capacity investment.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Scope

The model assumes exponential service, Poisson external arrivals, Markovian routing, infinite buffers, FCFS single-server nodes, and steady-state stability. Blocking, priorities, abandonment, finite buffers, general service distributions, and heavy-traffic diffusion approximations remain separate extensions.
