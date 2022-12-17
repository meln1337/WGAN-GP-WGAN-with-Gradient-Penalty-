import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty
from model import Critic, Generator, initialize_weights
from tqdm import tqdm

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.ImageFolder(root="/files/celeba", transform=transforms)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/real")
writer_fake = SummaryWriter(f"runs/fake")
step = 1

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    lossC_pe = 0
    lossG_pe = 0
    train_loader = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch: {epoch + 1}")
    for batch_idx, (real, _) in train_loader:
        real = real.to(device)
        BS = real.shape[0]
        fr = (BATCH_SIZE / BS) * len(loader)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for ci in range(CRITIC_ITERATIONS):
            noise = torch.randn(BS, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device=device)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Спробувати (torch.mean(critic_real) - torch.mean(critic_fake) + LAMBDA_GP * gp)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            if ci == CRITIC_ITERATIONS - 1:
                with torch.no_grad():
                    lossC_pe += loss_critic.item()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        with torch.no_grad():
            lossG_pe += loss_gen.item()

        # Print losses occasionally and print to tensorboard

        if ((batch_idx + 1) % 100) == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

        train_loader.set_postfix(loss_critic=fr * loss_critic.item(),
                                 loss_gen=fr * loss_gen.item())

    print(f"Epoch: {epoch+1}, Loss critic: {lossC_pe}, Loss genenerator: {lossG_pe}")
    writer_real.add_scalar("Loss critic", lossC_pe, global_step=epoch+1)
    writer_fake.add_scalar("Loss generator", lossG_pe, global_step=epoch+1)

torch.save(critic, "./critic.pth")
torch.save(gen, "./generator.pth")

print("Models are saved")