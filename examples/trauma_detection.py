from pathlib import Path
from PIL import Image

from llava_lens.core.config import Config
from llava_lens.core.registry import get_registry
from llava_lens.analysis.pipeline import AnalysisPipeline
from llava_lens.visualization.dashboard import DashboardGenerator


def main():
    config = Config()
    config.model.name = "llava-hf/llava-1.5-7b-hf"

    registry = get_registry()
    model_cls = registry.get("models", "llava")
    model = model_cls(config)
    model.load()

    pipeline = AnalysisPipeline(config, model)
    dashboard = DashboardGenerator(config)

    image_path = "test_image.jpg"
    prompt = "Describe any injuries or trauma visible in this image."

    if not Path(image_path).exists():
        print(f"Please provide an image at {image_path}")
        return

    result = pipeline.analyze(image_path, prompt)

    print(f"Model: {config.model.name}")
    print(f"High confidence patches: {len(result.patch_selection.high_confidence_indices)}")
    print(f"Low confidence patches: {len(result.patch_selection.low_confidence_indices)}")
    print(f"Selected patches: {len(result.patch_selection.selected_indices)}")

    dashboard.generate(
        image=image_path,
        logit_lens_result=result.logit_lens,
        prompt=prompt,
        model_name=config.model.name,
        output_path="trauma_analysis.html",
    )
    print("Dashboard saved to trauma_analysis.html")


if __name__ == "__main__":
    main()
