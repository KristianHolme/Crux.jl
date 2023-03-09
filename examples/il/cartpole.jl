using Debugger, Revise #remove
using Crux, Flux, POMDPGym, Random, POMDPs, BSON


## Cartpole
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)
γ = Float32(discount(mdp))

Disc() = ContinuousNetwork(Chain(Dense(6, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
V() = ContinuousNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = DiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
SA() = SoftDiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as;α=Float32(1.))


# Fill a buffer with expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/cartpole.bson")[:data]

# IQLearn ΔN, c_opt epochs hyperparams
mix = [(1,1),(1,5), (4,5), (4,10), (20,20), (20,50)]
𝒮_iqls = [OnlineIQLearn(π=SA(), 𝒟_demo=expert_trajectories, S=S, γ=γ, N=10000, ΔN=dn, log=(;period=100), c_opt=(;epochs=e)) for (dn,e) in mix]
[@time solve(i, mdp) for i in 𝒮_iqls]
p = plot_learning(𝒮_iqls, title = "CartPole-V0 IQL Tradeoff Curves", 
    labels = ["IQL ΔN=($dn),ep=($e)" for (dn,e) in mix])
Crux.savefig(p, "examples/il/cartpole_iqlearn_dne_tradeoffs.pdf")

# IQLearn λ_gp, α_reg hyperparams
λ_gps = Float32[1, 0.1, 0.01, 0.]
α_regs = Float32[100, 1, 0.1]
mix = Iterators.product(λ_gps,α_regs)
𝒮_iqls = [OnlineIQLearn(π=SA(), 𝒟_demo=expert_trajectories, 
    S=S, γ=γ, λ_gp=i, α_reg=j, N=10000, ΔN=1, 
    log=(;period=100), c_opt=(;epochs=1)) for (i,j) in mix]
[@time solve(i, mdp) for i in 𝒮_iqls]
p = plot_learning(𝒮_iqls, title = "CartPole-V0 IQL Tradeoff Curves", 
    labels = ["IQL λ_gp=($i),α_reg=($j)" for (i,j) in mix])
Crux.savefig(p, "examples/il/cartpole_iqlearn_reg_tradeoffs.pdf")

# Solve with PPO-GAIL
𝒮_gail = OnPolicyGAIL(D=Disc(), γ=γ, gan_loss=GAN_BCELoss(), 𝒟_demo=expert_trajectories, solver=PPO, π=ActorCritic(A(), V()), S=S, 
    N=40000, ΔN=1024, d_opt=(batch_size=1024, epochs=80))
solve(𝒮_gail, mdp)

# Solve with Behavioral Cloning
𝒮_bc = BC(π=A(), 𝒟_demo=expert_trajectories, S=S, opt=(epochs=100,), log=(period=10,))
solve(𝒮_bc, mdp)

# Solve with IQ-Learn
𝒮_iql = OnlineIQLearn(π=SA(), 𝒟_demo=expert_trajectories, S=S, γ=γ, N=40000, ΔN=200, log=(;period=50))
solve(𝒮_iql, mdp)

# Plot true rewards
p = plot_learning([𝒮_gail, 𝒮_bc, 𝒮_iql], title = "CartPole-V0 Imitation Training Curves", 
    labels = ["GAIL", "BC", "IQLearn"])
Crux.savefig(p, "examples/il/cartpole_training.pdf")