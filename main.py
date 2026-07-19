import sys
import os 

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py {train|sample} [args...]")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    

    if command == "train":
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from handler import UnifiedDataLoader 
        from data.prepare_tabular_data import save_tabular_jsonl, save_dataset_info, save_dataset_info_wtext
        
        loader = UnifiedDataLoader(dataset_name=sys.argv[2], target_model_type="tabdlm")
        meta = loader.get_metadata()
        data = {
            'train': loader.get_train_data(),
            'valid': loader.get_val_data(),
            'test': loader.get_test_data(), 
        }
        alldf = loader.get_all_data()
        
        save_tabular_jsonl(data, f'data/tabular/{meta["dataset_name"]}', target_col=meta['target'])
        if len(meta['text']) > 0:
            save_dataset_info_wtext(alldf, meta["dataset_name"], meta['type'], 
                                    meta['columns'],meta['nums'], meta['categorical'], 
                                    meta['text'], [meta['target']],
                                    f'data/tabular/{meta["dataset_name"]}')
        else:
            save_dataset_info(alldf, meta["dataset_name"], meta['type'], meta['columns'], 
                              meta['nums'], meta['categorical'], [meta['target']], 
                              f'data/tabular/{meta["dataset_name"]}')
        

        
        from tabdlm.cli.train import main as train_main
        train_main()
    elif command == "sample":
        from tabdlm.cli.sample import main as sample_main
        sample_main()
    else:
        print(f"Unknown command: {command!r}. Use 'train' or 'sample'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
