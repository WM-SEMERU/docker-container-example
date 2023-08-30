total_lines=$(wc -l < merged.txt)

part1_lines=$((total_lines * 70 / 100))
part2_lines=$((total_lines * 15 / 100))
part3_lines=$((total_lines * 15 / 100)) # This should be the remaining lines

head -n $part1_lines merged.txt > webtext_train.txt
tail -n +$(($part1_lines + 1)) merged.txt | head -n $part2_lines > webtext_valid.txt
tail -n +$(($part1_lines + $part2_lines + 1)) merged.txt > webtext_test.txt


