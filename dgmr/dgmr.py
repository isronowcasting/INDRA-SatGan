import pytorch_lightning as pl
import torch

# Import model components, not evaluation utils
from dgmr.common import ContextConditioningStack, LatentConditioningStack
from dgmr.discriminators import Discriminator
from dgmr.generators import Generator, Sampler
from dgmr.hub import NowcastingModelHubMixin
from dgmr.losses import GridCellLoss, NowcastingLoss, loss_hinge_disc, loss_hinge_gen

def weight_fn(y, precip_weight_cap=24.0):
    """Weight function for the grid cell loss, kept here as it's part of the loss logic."""
    return torch.max(y + 1, torch.tensor(precip_weight_cap, device=y.device))

class DGMR(pl.LightningModule, NowcastingModelHubMixin):
    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        visualize: bool = False,
        conv_type: str = "standard",
        grid_lambda: float = 5.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768,
        context_channels: int = 384,
        generation_steps: int = 6,
        num_samples: int = 6,
        precip_weight_cap: float = 24.0,
        input_frames: int = 6,
        **kwargs,
    ):
        super().__init__()
        # Using save_hyperparameters to make all args available via self.hparams
        self.save_hyperparameters()

        # Instantiate components
        self.conditioning_stack = ContextConditioningStack(input_channels=self.hparams.input_channels, conv_type=self.hparams.conv_type, output_channels=self.hparams.context_channels, num_context_steps=self.hparams.input_frames)
        self.latent_stack = LatentConditioningStack(shape=(8 * self.hparams.input_channels, self.hparams.output_shape // 32, self.hparams.output_shape // 32), output_channels=self.hparams.latent_channels)
        self.sampler = Sampler(forecast_steps=self.hparams.forecast_steps, latent_channels=self.hparams.latent_channels, context_channels=self.hparams.context_channels)
        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        self.discriminator = Discriminator(self.hparams.input_channels)
        self.grid_regularizer = GridCellLoss(weight_fn=weight_fn, precip_weight_cap=self.hparams.precip_weight_cap)
        
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        # Training step logic is unchanged and focused on optimization
        images, future_images = batch
        images, future_images = images.float(), future_images.float()
        g_opt, d_opt = self.optimizers()

        # Discriminator optimization
        for _ in range(2):
            d_opt.zero_grad()
            with torch.no_grad():
                predictions = self(images)
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(concatenated_outputs, images.size(0))
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            self.manual_backward(discriminator_loss)
            d_opt.step()

        # Generator optimization
        g_opt.zero_grad()
        predictions = [self(images) for _ in range(self.hparams.generation_steps)]
        gen_mean = torch.stack(predictions, dim=0).mean(dim=0)
        grid_cell_reg = self.grid_regularizer(gen_mean, future_images)
        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            _, score_generated = torch.split(concatenated_outputs, images.size(0))
            generated_scores.append(score_generated)
        
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.hparams.grid_lambda * grid_cell_reg
        self.manual_backward(generator_loss)
        g_opt.step()

        self.log_dict({"train/d_loss": discriminator_loss, "train/g_loss": generator_loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, future_images = batch
        images, future_images = images.float(), future_images.float()
        ensemble_predictions = [self(images) for _ in range(self.hparams.num_samples)]
        ensemble_tensor = torch.stack(ensemble_predictions, dim=0)
        mean_prediction = ensemble_tensor.mean(dim=0)
        return images, future_images, mean_prediction, ensemble_tensor

    def configure_optimizers(self):
        b1, b2 = self.hparams.beta1, self.hparams.beta2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.gen_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.disc_lr, betas=(b1, b2))
        return [opt_g, opt_d], []