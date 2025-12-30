use colored::*;

pub fn print_success(msg: &str) {
    println!("{} {}", "✓".green(), msg);
}

pub fn print_error(msg: &str) {
    eprintln!("{} {}", "✗".red(), msg);
}

pub fn print_info(msg: &str) {
    println!("{}", msg.cyan());
}

pub fn print_header(msg: &str) {
    println!("\n{}", msg.bold().underline());
}
