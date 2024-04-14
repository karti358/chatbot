use std::any::type_name_of_val;

fn main () {
    let vec: Vec<String> = Vec::from([
        String::from("1"), 
        String::from("22"), 
        String::from("3")
    ]);
    let comp = vec.join(",");
    println!("{}", type_name_of_val(&comp));
}