from haystack.components.generators import OpenAIGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from haystack.components.builders import PromptBuilder
from haystack import Pipeline
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv(override=True)


class LabelGenerator:
    def __init__(self, model_name=None, max_workers=4):
        self.max_workers = max_workers
        self.generator = OpenAIGenerator(model="gpt-4.1-nano")
        self.templates = """
        CONDITION:
        - Front doors: left={{front_left_door}}, right={{front_right_door}}
        - Rear doors: left={{rear_left_door}}, right={{rear_right_door}}
        - Hood: {{hood}}

        TASK:
        Describe this in natural language, mixing up your descriptions to create variety while staying accurate.

        IMPORTANT:
        - Only the description should be generated, and should be started by, The car's ....
        - Only use the word (closed or open) for explaining each condition.
        - Don't add explanations except for the conditions of doors and hood."""

        self.prompt_builder = PromptBuilder(
            template=self.templates,
            required_variables=[
                "front_left_door",
                "front_right_door",
                "rear_left_door",
                "rear_right_door",
                "hood",
            ],
        )

        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("llm", self.generator)
        self.pipeline.connect("prompt_builder", "llm")

    def generate_single(self, message: dict):
        result = self.pipeline.run(
            {
                "prompt_builder": {
                    "front_left_door": message["front_left_door"],
                    "front_right_door": message["front_right_door"],
                    "rear_left_door": message["rear_left_door"],
                    "rear_right_door": message["rear_right_door"],
                    "hood": message["hood"],
                }
            }
        )

        return result["llm"]["replies"][0].strip()

    def process_batch_concurrent(
        self, batch_data: List[Dict], batch_idx: int
    ) -> List[Dict]:
        results = []

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(batch_data))
        ) as executor:
            future_to_data = {}
            for i, row_data in enumerate(batch_data):
                future = executor.submit(self.generate_single, row_data["message"])
                future_to_data[future] = row_data

            for future in as_completed(future_to_data):
                row_data = future_to_data[future]
                try:
                    label = future.result(timeout=30)
                    results.append({"imagepath": row_data["imagepath"], "label": label})
                except Exception as e:
                    print(f"Error processing {row_data['imagepath']}: {e}")
                    results.append(
                        {
                            "imagepath": row_data["imagepath"],
                            "label": f"Error: {str(e)}",
                        }
                    )

        return results

    def process_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 50,
        save_interval: int = 500,
        output_file: str = "generated_labels.csv",
    ) -> pd.DataFrame:
        all_data = []
        print(f"Processing {len(df)} rows in batches of {batch_size}")

        for idx, row in df.iterrows():
            message = {
                "front_left_door": "open" if row["front_left_door"] == 1 else "closed",
                "front_right_door": "open"
                if row["front_right_door"] == 1
                else "closed",
                "rear_left_door": "open" if row["rear_left_door"] == 1 else "closed",
                "rear_right_door": "open" if row["rear_right_door"] == 1 else "closed",
                "hood": "open" if row["hood"] == 1 else "closed",
            }
            all_data.append(
                {
                    "imagepath": f"{row['filename']}",
                    "message": message,
                    "original_idx": idx,
                }
            )

        all_results = []
        total_batches = (len(all_data) + batch_size - 1) // batch_size
        start_time = time.time()

        for batch_idx in range(0, len(all_data), batch_size):
            batch_data = all_data[batch_idx : batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1

            print(
                f"\nProcessing batch {current_batch}/{total_batches} ({len(batch_data)} items)"
            )
            batch_start = time.time()

            batch_results = self.process_batch_concurrent(batch_data, batch_idx)
            all_results.extend(batch_results)

            batch_time = time.time() - batch_start
            total_time = time.time() - start_time
            avg_time_per_item = total_time / len(all_results)
            remaining_items = len(all_data) - len(all_results)
            eta = remaining_items * avg_time_per_item

            print(f"✓ Batch {current_batch} completed in {batch_time:.1f}s")
            print(
                f"Progress: {len(all_results)}/{len(all_data)} ({100 * len(all_results) / len(all_data):.1f}%)"
            )
            print(f"ETA: {eta / 60:.1f} minutes")

            if len(all_results) % save_interval == 0 or len(all_results) == len(
                all_data
            ):
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(f"temp_{output_file}", index=False)
                print(f"💾 Progress saved to temp_{output_file}")

        result_df = pd.DataFrame(all_results)
        result_df.to_csv(output_file, index=False)

        total_time = time.time() - start_time
        print("\n🎉 Processing completed!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Average time per item: {total_time / len(all_results):.2f} seconds")
        print(f"Results saved to {output_file}")

        return result_df

    def resume_from_checkpoint(
        self,
        df: pd.DataFrame,
        checkpoint_file: str,
        output_file: str = "generated_labels.csv",
    ) -> pd.DataFrame:
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            processed_files = set(checkpoint_df["imagepath"].tolist())

            remaining_df = df[~df["filename"].isin(processed_files)].copy()

            print(f"Found checkpoint with {len(checkpoint_df)} processed items")
            print(f"Remaining to process: {len(remaining_df)} items")

            if len(remaining_df) == 0:
                print("All items already processed!")
                return checkpoint_df

            new_results_df = self.process_dataframe(
                remaining_df, output_file="new_results.csv"
            )

            combined_df = pd.concat([checkpoint_df, new_results_df], ignore_index=True)
            combined_df.to_csv(output_file, index=False)

            return combined_df

        except FileNotFoundError:
            print(f"Checkpoint file {checkpoint_file} not found. Starting fresh.")
            return self.process_dataframe(df, output_file=output_file)


if "__main__" == __name__:
    label_generator = LabelGenerator(max_workers=3)
    df = pd.read_csv("../preparation/dataset/metadata.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Loaded dataset with {len(df)} rows")

    generated_labels = []

    result_df = label_generator.process_dataframe(
        df,
        batch_size=25,
        save_interval=25,
        output_file="generated_labels.csv",
    )

    result_df.head()
