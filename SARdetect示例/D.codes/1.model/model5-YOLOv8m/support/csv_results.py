"""Offline stdlib equivalent of the optional Polars results CSV loader."""
def install_csv_reader():
    import csv
    from ultralytics.engine.trainer import BaseTrainer
    def read_results_csv(self):
        if not self.csv.is_file():return {}
        with self.csv.open(newline='') as stream:
            reader=csv.DictReader(stream);result={k:[] for k in reader.fieldnames or []}
            for row in reader:
                for key,value in row.items():
                    if value is None:result[key].append(None);continue
                    try:value=float(value)
                    except ValueError:pass
                    result[key].append(value)
        return result
    BaseTrainer.read_results_csv=read_results_csv
