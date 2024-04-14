package main

// import (
//     "fmt"
//     // "errors"
// )
// // import "unicode/utf8"

// // func printMe(value string) {
// //     fmt.Println(value)
// // }
// // func divide(num int, den int) (int, int, error) {
// //     if den == 0 {
// //         err := errors.New("Cannot divide by Zero")
// //         return 0, 0, err
// //     }
// //     return num / den, num % den
// // }

// func main() {
//     //1. Variables, normal operations allowed, --, ++, -=, += etc.
//     // var a int32 = 10
//     // fmt.Println(a)

//     // var b float32 = 3.1234
//     // fmt.Println(b)

//     // var c float32 = 10.1
//     // var d int32 = 2
//     // var result float32 = c + float32(d)
//     // fmt.Println(result)

//     // var a int = 3
//     // var b int = 3
//     // fmt.Println(a / b) // integer division returns integer

//     // var a string = "Hello \n World" + "Heyy"
//     // fmt.Println(a)

//     // fmt.Println(len("Γ")) // does not print 1 since the len gives length of the utf encoding
//     // //Instead use
//     // fmt.Println(utf8.RuneCountInString("Γ"))

//     //Characters
//     // var a rune = 'a'
//     // fmt.Println(a)

//     //defaults values for all ints, floats, rune, is 0 and for string is "" and for bool is false

//     // a := "text"
//     // b, c := 1, 2

//     // const a string = "const value"
//     // fmt.Println(a)

//     //2. Functions and Control Structures

//     // var err error; //default value is nil
//     // res, rem, err := divide(1, 0)
//     // if err == nil{
//     //     fmt.Println("The dividend is %v with remainder %v", res, rem)
//     // }

//     //switch statement
//     // switch {
//     //     case err != nil:
//     //         fmt.Println(err.Error())
//     //         //dont need explicit break statements
//     //     case rem == 0:
//     //         fmt.Println("The remainder is %v", rem)
//     // }

//     // conditional switch statements
//     // switch rem {
//     //     case 1:
//     //     fmt.Println("Remainder is %v", rem)
//     //     default:
//     //     fmt.Println("Remainder is %v", rem)
//     // }

//     //3. Arrays, Slices, Maps, and Loops
//     //Arrays

//     // var arr [3]int32 = [3]int32{1,2,3} // or arr := [...]int32{1,2,3}
//     // arr[2] = 1234
//     // fmt.Println(arr, arr[1], &arr, &arr[0], &arr[1])

//     // slices are wrappers aroung arrays
//     // var arr []int32 = []int32{1,2,3}
//     // fmt.Println(len(arr), cap(arr))
//     // arr = append(arr, 7)
//     // fmt.Println(len(arr), cap(arr))

//     // var arr2 []int32 = []int32{4,5,6}
//     // arr = append(arr, arr2...)
//     // fmt.Println(arr)
//     // // The three dots (…) in Golang is termed as Ellipsis in Golang which is used in the variadic function. The function that called with the varying number of arguments is known as variadic function. Or in other words, a user is allowed to pass zero or more arguments in the variadic function.
//     // // The last parameter of the variadic function is always using the Ellipsis. It means it can accept any number of arguments.

//     // var arr3 []int32 = make(int32[], 3, 8) // specifying capacity makes slice not allocate memory dynamically when we append elements

//     // Maps
//     // always a return a value for a key, even if not exists
//     // also return a boolean whether key exists, this can be used
//     // remove a element by delete

//     var m map[string]int32 = map[string]int32{"Adam":1, "Russ":3}
//     delete(m, "Adam")
//     fmt.Println(m)
//     var age, ok = m["Hey"]
//     if ok {
//         fmt.Println("Hey exists, and " + string(age))
//     } else {
//         fmt.Println("Hey does not exist")
//     }

//     //Loops
//     // has only for Loop, many types
//     for name, age := range m {

//         fmt.Println("a ", name, age)
//     }

//     for i, v := range []int32{1,2,3,4,5} {
//         fmt.Println("b ", i, v)
//     }

//     i := 0
//     for i < 10 {
//         fmt.Println("c ", i)
//         i += 1
//     }

//     i = 0
//     for {
//         if i == 10 {break}
//         fmt.Println("d ", i)
//         i += 1
//     }

//     for i:= 0; i < 10; i++ {
//         fmt.Println("e ", i)
//     }

//     //4. Strings, runes, bytes
//     // strings are stored as utf encodings

// }

//7. goroutines
//concurrency, rarely multithreading
// package main

// import (
//     "fmt"
//     "time"
//     "sync"
// )

// var wg = sync.WaitGroup{}
// var m = sync.Mutex{}
// var rm = sync.RWMutex{}
// var dbData = []string {"asf", "sdf", "we", "dfsd"}
// var results = []string{}

// func main() {
//     t0 := time.Now()

//     for i:=0; i < len(dbData); i++ {
//         wg.Add(1)
//         go dbCall(i)
//     }
//     wg.Wait()
//     fmt.Println(time.Since(t0))
//     fmt.Println(results) // the may not have proper values as in dbData, this shows concurrent execution of tasks
// }
// func dbCall(i int) {
//     var delay float32 = 2000
//     time.Sleep(time.Duration(delay) * time.Millisecond)
//     fmt.Println(dbData[i])
//     save(dbData[i])
//     log()
//     wg.Done()
// }

// func save(result string) {
//     rm.Lock()
//     results = append(results, result)
//     rm.Unlock()
// }

// func log() {
//     rm.RLock()
//     fmt.Println(results)
//     rm.RUnlock()
// }

//8. Channels
// Hold data, thread safe, listen for data

// package main
// import (
//     "fmt"
//     "time"
// )

// func main() {
//     var c = make(chan int, 5) // kind of underlying array
//     go process(c)
//     for i := range c {
//         fmt.Println(i , <- c)
//         time.Sleep(time.Second + 1)
//     }
// }
// func process(c chan int) {
//     defer close(c) // do at the exit of function
//     for i := 0; i < 5; i++{
//         c <- i
//     }
//     fmt.Println("Exiting process")
// }

// 9. generics
// package main
// import "fmt"

// type car [T int | float32] struct {
// 	number T
// 	plate T
// }

// func main() {
//     var slice = []int{1,2,3,4}
//     fmt.Println(sum_slice[int](slice))

// 	fmt.Println(isEmpty([]float32{1.4, 2.3, 3.4}))

// 	var s = car[int]{12, 13}
// 	fmt.Println(s)
// }

// func sum_slice[T int | float32 | float64] (slice []T) T {
//     var sum T
//     for _, v := range slice {
//         sum += v
//     }
//     return sum
// }

// func isEmpty[T any] (slice []T) bool {
// 	return len(slice) == 0
// }
