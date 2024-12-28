import argparse
import logging
import logging.config
import os

# Set up logging
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
logger = logging.getLogger("SimpleTuner")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
from helpers.training.trainer import Trainer
from helpers.training.state_tracker import StateTracker
from helpers import log_format

if __name__ == "__main__":
    # Define command-line arguments
    # parser = argparse.ArgumentParser(description="Train a model with SimpleTuner.")
    #
    # # Add your arguments here
    # parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # parser.add_argument("--data_backend_config", type=str, required=True)
    # parser.add_argument("--aspect_bucket_rounding", type=int, default=2)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--minimum_image_size", type=int, default=0)
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--lora_type", type=str, default="lycoris")
    # parser.add_argument("--lycoris_config", type=str, required=True)
    # parser.add_argument("--max_train_steps", type=int, default=10000)
    # parser.add_argument("--num_train_epochs", type=int, default=0)
    # parser.add_argument("--checkpointing_steps", type=int, default=1000)
    # parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    # parser.add_argument("--hub_model_id", type=str, default="simpletuner-lora")
    # parser.add_argument("--push_to_hub", type=str, default="false")
    # parser.add_argument("--push_checkpoints_to_hub", type=str, default="false")
    # parser.add_argument("--tracker_project_name", type=str, default="lora-training")
    # parser.add_argument("--tracker_run_name", type=str, default="simpletuner-lora")
    # parser.add_argument("--report_to", type=str, default="tensorboard")
    # parser.add_argument("--model_type", type=str, default="lora")
    # parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    # parser.add_argument("--model_family", type=str, default="flux")
    # parser.add_argument("--train_batch_size", type=int, default=1)
    # parser.add_argument("--gradient_checkpointing", type=str, default="true")
    # parser.add_argument("--caption_dropout_probability", type=float, default=0.1)
    # parser.add_argument("--resolution_type", type=str, default="pixel_area")
    # parser.add_argument("--resolution", type=int, default=1024)
    # parser.add_argument("--validation_seed", type=int, default=42)
    # parser.add_argument("--validation_steps", type=int, default=1000)
    # parser.add_argument("--validation_resolution", type=str, default="1024x1024")
    # parser.add_argument("--validation_guidance", type=float, default=3.0)
    # parser.add_argument("--validation_guidance_rescale", type=str, default="0.0")
    # parser.add_argument("--validation_num_inference_steps", type=int, default=20)
    # parser.add_argument("--validation_prompt", type=str, default="")
    # parser.add_argument("--mixed_precision", type=str, default="bf16")
    # parser.add_argument("--optimizer", type=str, default="adamw_bf16")
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    # parser.add_argument("--lr_scheduler", type=str, default="polynomial")
    # parser.add_argument("--lr_warmup_steps", type=int, default=100)
    # parser.add_argument("--validation_torch_compile", type=str, default="false")
    # parser.add_argument("--disable_benchmark", type=str, default="false")
    # parser.add_argument("--base_model_precision", type=str, default="int8-quanto")
    # parser.add_argument("--text_encoder_1_precision", type=str, default="no_change")
    # parser.add_argument("--text_encoder_2_precision", type=str, default="no_change")
    # parser.add_argument("--lora_rank", type=int, default=16)
    # parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # parser.add_argument("--base_model_default_dtype", type=str, default="bf16")
    # parser.add_argument("--user_prompt_library", type=str, required=True)
    #
    # # Parse arguments
    # args = parser.parse_args()
    #
    # # Convert to dictionary
    # config = vars(args)
    # print("Parsed Config:", config)
    #
    # # Use the config dictionary in your trainer
    try:
        trainer = Trainer(
            config={"resume_from_checkpoint": "latest", "data_backend_config": "config/dataset.hilary.json",
                    "aspect_bucket_rounding": 2, "seed": 42, "minimum_image_size": 0, "output_dir": "output/models",
                    "lora_type": "lycoris", "lycoris_config": "config/lycoris_config.json", "max_train_steps": 10000,
                    "num_train_epochs": 0, "checkpointing_steps": 1000, "checkpoints_total_limit": 5,
                    "hub_model_id": "simpletuner-lora", "push_to_hub": False, "push_checkpoints_to_hub": False,
                    "tracker_project_name": "lora-training", "tracker_run_name": "simpletuner-lora",
                    "report_to": "tensorboard", "model_type": "lora",
                    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev", "model_family": "flux",
                    "train_batch_size": 1, "gradient_checkpointing": True, "caption_dropout_probability": 0.1,
                    "resolution_type": "pixel_area", "resolution": 1024, "validation_seed": 42,
                    "validation_steps": 1000, "validation_resolution": "1024x1024", "validation_guidance": 3.0,
                    "validation_guidance_rescale": 0.0, "validation_num_inference_steps": 20,
                    "validation_prompt": "hilary duff a young woman standing on a beach wearing a white shirt,looking at the sea and smiling,behind her are a few people,sand,and a blue sky",
                    "mixed_precision": "bf16", "optimizer": "adamw_bf16", "learning_rate": 1e-4,
                    "lr_scheduler": "polynomial", "lr_warmup_steps": 100, "validation_torch_compile": False,
                    "disable_benchmark": False, "base_model_precision": "int8-quanto",
                    "text_encoder_1_precision": "no_change", "text_encoder_2_precision": "no_change", "lora_rank": 16,
                    "max_grad_norm": 1.0, "base_model_default_dtype": "bf16",
                    "user_prompt_library": "config/user_prompt_library.json"},
            exit_on_error=True,
        )
        trainer.configure_webhook()
        trainer.init_noise_schedule()
        trainer.init_seed()

        trainer.init_huggingface_hub()

        trainer.init_preprocessing_models()
        trainer.init_precision(preprocessing_models_only=True)
        trainer.init_data_backend()
        # trainer.init_validation_prompts()
        trainer.init_unload_text_encoder()
        trainer.init_unload_vae()

        trainer.init_load_base_model()
        trainer.init_controlnet_model()
        trainer.init_precision()
        trainer.init_freeze_models()
        trainer.init_trainable_peft_adapter()
        trainer.init_ema_model()
        # EMA must be quantised if the base model is as well.
        trainer.init_precision(ema_only=True)

        trainer.move_models(destination="accelerator")
        trainer.init_validations()
        trainer.init_benchmark_base_model()

        trainer.resume_and_prepare()

        trainer.init_trackers()
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
