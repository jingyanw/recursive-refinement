function str = append_c(str, category)
% APPEND_C: Helper function to append a number at the end of a string.

str = [str '_' num2str(category)];
