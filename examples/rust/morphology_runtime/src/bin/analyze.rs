use morphology_runtime::api::MorphologyRuntime;

fn main() {
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let mut model_path = format!("{}/resources/model.onnx", crate_dir);
    let mut res_dir = format!("{}/resources", crate_dir);

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() { model_path = args[i+1].clone(); args.drain(i..=i+1); continue; }
        if args[i] == "--resources" && i + 1 < args.len() { res_dir = args[i+1].clone(); args.drain(i..=i+1); continue; }
        i += 1;
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


