use chrono::Utc;
use serde::Serialize;
use serde_json::json;
use std::fs::OpenOptions;
use std::{collections::HashMap, io::Write};

#[derive(Serialize)]
struct LogEntry<T> {
    value: T,
    epoch: usize,
    exec_date: String,
    entry_type: String,
    tags: HashMap<String, String>,
}

pub struct Logger {
    path: String,
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger {
    pub fn new() -> Self {
        let now = Utc::now().to_rfc3339();
        let path = format!("logs/{}", now);
        std::fs::create_dir_all(&path).expect("unable to create logs directory");
        Self { path }
    }

    pub fn errors(&mut self, epoch: usize, id: usize, layer: &str, errors: &[f64]) {
        let entry = LogEntry {
            epoch,
            value: errors.to_vec(),
            entry_type: "error".to_string(),
            exec_date: Utc::now().to_rfc3339(),
            tags: HashMap::from([
                ("id".to_string(), id.to_string()),
                ("layer".to_string(), layer.to_string()),
            ]),
        };
        self.log(entry);
    }

    pub fn loss(&mut self, epoch: usize, id: usize, loss: f64) {
        let entry = LogEntry {
            epoch,
            value: loss,
            entry_type: "loss".to_string(),
            exec_date: Utc::now().to_rfc3339(),
            tags: HashMap::from([("id".to_string(), id.to_string())]),
        };
        self.log(entry);
    }

    pub fn accuracy(&mut self, epoch: usize, id: usize, accuracy: f64) {
        let entry = LogEntry {
            epoch,
            value: accuracy,
            entry_type: "accuracy".to_string(),
            exec_date: Utc::now().to_rfc3339(),
            tags: HashMap::from([("id".to_string(), id.to_string())]),
        };
        self.log(entry);
    }

    pub fn gradients(&mut self, epoch: usize, id: usize, layer: &str, gradients: &[f64]) {
        let entry = LogEntry {
            epoch,
            value: gradients.to_vec(),
            entry_type: "gradient".to_string(),
            exec_date: Utc::now().to_rfc3339(),
            tags: HashMap::from([
                ("id".to_string(), id.to_string()),
                ("layer".to_string(), layer.to_string()),
            ]),
        };
        self.log(entry);
    }

    pub fn weights(&mut self, epoch: usize, id: usize, layer: &str, weights: Vec<Vec<f64>>) {
        let entry = LogEntry {
            epoch,
            value: weights,
            entry_type: "weights".to_string(),
            exec_date: Utc::now().to_rfc3339(),
            tags: HashMap::from([
                ("id".to_string(), id.to_string()),
                ("layer".to_string(), layer.to_string()),
            ]),
        };
        self.log(entry);
    }

    fn log<T: Serialize>(&mut self, entry: LogEntry<T>) {
        let file_name = format!("{}/{}.jsonl", self.path, entry.entry_type);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_name)
            .expect("unable to open log file");

        writeln!(file, "{}", json!(entry)).expect("Unable to write to log file");
    }
}
