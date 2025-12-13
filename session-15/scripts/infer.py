import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from session_15 import UNetModule


def load_model(checkpoint_path: str) -> UNetModule:
    model = UNetModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def load_and_preprocess_image(
    image_path: str, image_size: tuple[int, int] = (256, 256)
) -> tuple[torch.Tensor, Image.Image]:
    # Load original image
    original_img = Image.open(image_path).convert("RGB")

    # Preprocess for model
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(original_img).unsqueeze(0)  # type: ignore

    return img_tensor, original_img


def predict_mask(
    model: UNetModule,
    image_tensor: torch.Tensor,
    original_size: tuple[int, int],
    device: str,
) -> torch.Tensor:
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    # Resize mask to original image size
    # original_size is (width, height), but resize expects (height, width)
    mask_resized = transforms.functional.resize(  # type: ignore
        mask,
        size=(original_size[1], original_size[0]),  # (H, W)
        interpolation=InterpolationMode.NEAREST,
    )

    return mask_resized.squeeze().cpu()


def apply_mask_to_image(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    img_array = np.array(image)
    mask_array = mask.numpy()

    # Create RGB mask (broadcast mask to 3 channels)
    masked_img = img_array * mask_array[:, :, np.newaxis]

    return Image.fromarray(masked_img.astype(np.uint8))


def visualize_prediction(
    original_img: Image.Image,
    predicted_mask: torch.Tensor,
    save_path: str,
):
    # Calculate aspect ratio and determine figure size
    img_width, img_height = original_img.size
    aspect_ratio = img_width / img_height

    # Base height for the figure
    base_height = 10

    # Adjust width based on aspect ratio
    # For 2 columns, multiply by 2
    fig_width = base_height * aspect_ratio * 2
    fig_height = base_height * 2  # 2 rows

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Predicted mask
    axes[0, 1].imshow(predicted_mask, cmap="gray")
    axes[0, 1].set_title("Predicted Mask")
    axes[0, 1].axis("off")

    # Masked image (foreground only)
    masked_img = apply_mask_to_image(original_img, predicted_mask)
    axes[1, 0].imshow(masked_img)
    axes[1, 0].set_title("Masked Image")
    axes[1, 0].axis("off")

    # Overlay
    axes[1, 1].imshow(original_img)
    axes[1, 1].imshow(predicted_mask, alpha=0.5, cmap="jet")
    axes[1, 1].set_title("Overlay")
    axes[1, 1].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def main(
    checkpoint_path: str,
    images: list[str],
    output_dir: Path,
    image_size: int,
    device: str,
):
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path)
    model = model.to(device)

    # Process each image
    for img_path in images:
        print(f"\nProcessing {img_path}")

        # Load and preprocess
        img_tensor, original_img = load_and_preprocess_image(
            img_path, image_size=(image_size, image_size)
        )

        # Predict (mask will be resized to original dimensions)
        predicted_mask = predict_mask(model, img_tensor, original_img.size, device)

        # Save visualization
        img_name = Path(img_path).stem
        save_path = output_dir / f"{img_name}_prediction.png"
        visualize_prediction(original_img, predicted_mask, str(save_path))

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer segmentation masks using trained UNet"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--images", type=str, nargs="+", required=True, help="Paths to input images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size (square)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()
    main(
        args.checkpoint,
        args.images,
        args.output_dir,
        args.image_size,
        args.device,
    )
