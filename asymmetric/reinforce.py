####### Remove value head and use same score for all tokens
from trl.core import (
    masked_whiten,
    masked_mean,
    masked_var,
    entropy_from_logits,
    flatten_dict,
)
import torch
from trl import PPOTrainer
class REINFORCETrainer(PPOTrainer):
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # no KL shaping at all
            kls.append(torch.zeros_like(logprob))
            non_score_reward = torch.zeros_like(logprob)

            reward = torch.zeros_like(logprob)
            reward[mask.bool()] = score  # SAME scalar on every response token
            rewards.append(reward)
            non_score_rewards.append(non_score_reward)

        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def compute_advantages(self, values, rewards, mask):
        # No baseline, no discounting/smoothing, no whitening
        values = torch.zeros_like(rewards)
        advantages = rewards.detach() * mask
        returns = advantages
        return values, advantages, returns

    def loss(
        self,
        old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
    ):
        # standard PPO clipping around the REINFORCE advantage
        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses  = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.config.cliprange,
                                               1.0 + self.config.cliprange)
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)

        # no value loss at all
        entropy = masked_mean(entropy_from_logits(logits), mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=torch.tensor(0.0, device=pg_loss.device), total=pg_loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=0.5 * masked_mean((logprobs - old_logprobs)**2, mask).detach(),
                policykl=masked_mean(old_logprobs - logprobs, mask).detach(),
                clipfrac=masked_mean((pg_losses2 > pg_losses).float(), mask).detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=masked_mean(returns, mask).detach(), var=masked_var(returns, mask).detach()),
            val=dict(vpred=torch.tensor(0.0, device=pg_loss.device), error=torch.tensor(0.0, device=pg_loss.device),
                     clipfrac=torch.tensor(0.0, device=pg_loss.device),
                     mean=torch.tensor(0.0, device=pg_loss.device), var=torch.tensor(0.0, device=pg_loss.device)),
        )
        # Return (policy loss, 0 * value loss)
        return pg_loss, torch.tensor(0.0, device=pg_loss.device), flatten_dict(stats)
