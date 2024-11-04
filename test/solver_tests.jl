using POMDPs
using POMDPGym
import POMDPModels
using Test
using Crux
using Flux
using Random
using BSON

function test_solver(𝒮fn, mdp, π...)
    # run it once
    Random.seed!(0)
    S1 = 𝒮fn(deepcopy.(π)...)
    π1 = solve(S1, deepcopy(mdp))

    # run it again on the cpu with the same rng
    Random.seed!(0)
    S2 = 𝒮fn(deepcopy.(π)...)
    π2 = solve(S2, deepcopy(mdp))

    # Run it on the gpu
    Random.seed!(0)
    S3 = 𝒮fn(gpu.(deepcopy.(π))...)
    π3 = solve(S3, deepcopy(mdp))

    # compare the results
    s = rand(Crux.dim(state_space(mdp))...)
    try
        value(π[1], s)
        @test all(value(π1, s) .≈ value(π2, s))
        @test all(abs.(value(π2, s) .- value(π3, s)) .< 1e-2)
    catch
        if action(π1, s)[1] isa Symbol
            @test all(action(π1, s) .== action(π2, s))
            @test all(action(π2, s) .== action(π3, s))
        else
            @test isapprox(action(π1, s), action(π2, s), atol=1e-1)
            @test isapprox(action(π2, s), action(π3, s), atol=1e-1)
            @test all(abs.(action(π2, s) .- action(π3, s)) .< 1e-2)
        end
    end
end

## Training params
N = 100
ΔN = 50

## discrete RL
discrete_mdp = GridWorldMDP()
S = state_space(discrete_mdp)
A() = DiscreteNetwork(Chain(Dense(2, 32, relu), Dense(32, 4)), actions(discrete_mdp))
V() = ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1)))
AC() = ActorCritic(A(), V())

test_solver((π) -> REINFORCE(π=π, S=S, N=N, ΔN=ΔN), discrete_mdp, A())
test_solver((π) -> A2C(π=π, S=S, N=N, ΔN=ΔN), discrete_mdp, AC())
test_solver((π) -> PPO(π=π, S=S, N=N, ΔN=ΔN), discrete_mdp, AC())
test_solver((π) -> DQN(π=π, S=S, N=N), discrete_mdp, A())

# test compatibility with non-POMDPGym MDPs
test_solver((π) -> PPO(π=π, S=S, N=N, ΔN=ΔN), POMDPModels.SimpleGridWorld(), AC())

## Continuous RL
continuous_mdp = PendulumPOMDP()
S = state_space(continuous_mdp)
QSA() = ContinuousNetwork(Chain(Dense(3, 32, tanh), Dense(32, 1)))
V() = ContinuousNetwork(Chain( Dense(2, 32, relu), Dense(32, 1)))
A() = ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh)), 1)
G() = GaussianPolicy(A(), zeros(Float32, 1))

test_solver((π) -> REINFORCE(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, G())
test_solver((π) -> A2C(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), V()))
test_solver((π) -> PPO(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), V()))
test_solver((π) -> DDPG(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(A(), QSA()))
test_solver((π) -> TD3(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(A(), DoubleNetwork(QSA(), QSA())))
test_solver((π) -> SAC(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())))


# Continuous IL
γ = 0.95f0
exp_data_path = abspath(joinpath(@__DIR__, "..", "examples", "il", "expert_data", "pendulum.bson"))
𝒟_demo = expert_trajectories = BSON.load(exp_data_path)[:data]
D(output=1) = ContinuousNetwork(Chain(DenseSN(3, 12, relu), DenseSN(12, output)))

test_solver((π, D) -> OnPolicyGAIL(D=D, 𝒟_demo=𝒟_demo, π=π, S=S, N=N, ΔN=ΔN, γ=γ), continuous_mdp, ActorCritic(G(), V()), D())
test_solver((π, D) -> OffPolicyGAIL(D=D, 𝒟_demo=𝒟_demo, π=π, S=S, N=50, ΔN=ΔN), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())), D(2))
test_solver(π -> BC(π=π, 𝒟_demo=𝒟_demo, S=S, opt=(epochs=1,), log=(period=50,)), continuous_mdp, A())
# NOTE: gradient penalty on the gpu only plays nicely with tanh, not relus in the discriminator?
test_solver((π) -> AdVIL(𝒟_demo=𝒟_demo, π=π, S=S, a_opt=(epochs=1,), log=(period=50,)), continuous_mdp, ActorCritic(A(), QSA()))
test_solver((π) -> SQIL(𝒟_demo=𝒟_demo, π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())))
test_solver((π) -> AdRIL(𝒟_demo=𝒟_demo, π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())))
test_solver((π) -> ASAF(𝒟_demo=𝒟_demo, π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, G())


# Batch RL
test_solver(π -> BatchSAC(π=π, 𝒟_train=𝒟_demo, S=S, a_opt=(epochs=1,)), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())))
test_solver(π -> CQL(π=π, 𝒟_train=𝒟_demo, S=S, a_opt=(epochs=1,)), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())))
