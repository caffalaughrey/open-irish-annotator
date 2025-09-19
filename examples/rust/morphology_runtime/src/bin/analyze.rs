use morphology_runtime::api::MorphologyRuntime;
use std::path::Path;

fn main() {
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let mut model_path = format!("{crate_dir}/resources/model.onnx");
    let mut res_dir = format!("{crate_dir}/resources");

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() { model_path = args[i+1].clone(); args.drain(i..=i+1); continue; }
        if args[i] == "--resources" && i + 1 < args.len() { res_dir = args[i+1].clone(); args.drain(i..=i+1); continue; }
        i += 1;
    }

    // Fallback search to de-duplicate resources
    if !Path::new(&model_path).exists() {
        let alt = "artifacts/onnx/model.onnx";
        if Path::new(alt).exists() {
            model_path = alt.to_string();
        }
    }
    if !Path::new(&format!("{res_dir}/tagset.json")).exists() {
        let alt = "data/processed";
        if Path::new(&format!("{alt}/tagset.json")).exists() {
            res_dir = alt.to_string();
        }
    }

    let runtime = MorphologyRuntime::new_from_resources(&model_path, &res_dir)
        .expect("failed to init runtime");

    if args.is_empty() {
        use std::io::{self, Read};
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).expect("read stdin");
        for line in buf.lines() {
            let toks: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if toks.is_empty() { continue; }
            let analyses = runtime.analyze(toks).expect("analyze failed");
            for a in analyses { println!("{}\t{}\t{}", a.token, a.tag, a.lemma); }
        }
    } else {
        let analyses = runtime.analyze(args).expect("analyze failed");
        for a in analyses { println!("{}\t{}\t{}", a.token, a.tag, a.lemma); }
    }
}


