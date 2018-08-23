%baseDir = 'C:\Users\PuShanWen\Desktop\DMC\test\';
baseDir = '/Users/daojing/Desktop/DataMiningLAB/train';
contains = dir(baseDir);
feature = zeros((length(contains)-2)*500,19);
tmp = zeros(500,19);
for k = 3:length(contains)%ignore file:'.' and '..'
    curr_dir = dir([contains(k).folder,'\',contains(k).name]);
    class = k-2;
    for l =3:length(curr_dir)
        %load file
        fid = fopen([curr_dir(l).folder,'\',curr_dir(l).name]);
        str = fread(fid,'*char')';
        fclose(fid);
        
        %%regular expression
        %io count
        scanf_count = length(regexp(str,'scanf('));
        printf_count = length(regexp(str,'printf('));
        cin_count = length(regexp(str,'cin '));
        cout_count = length(regexp(str,'cout '));
        input_count = scanf_count + cin_count;
        output_count = printf_count + cout_count;
        
        %io token count
        dotf_count = length(regexp(str,'"*%.*f*"'));
        f_count = length(regexp(str,'%f'));
        d_count = length(regexp(str,'%d'));
        s_count = length(regexp(str,'%s'));
        lf_count = length(regexp(str,'%lf'));
        
        %array count
        dereference_count = length(regexp(str,'\*('));
        bracket_count = length(regexp(str,'\[*]'));
        %seri_count = regexp(str,'(?<=\[)(.*)(?=\])');
        reference_count = dereference_count + bracket_count;
        %dereference_count = length(regexp(str,'\*(*\*('));
        antibracket_count = length(regexp(str,']\['));
        
        %branch count
        if_count = length(regexp(str,'if '));
        switch_count = length(regexp(str,'switch '));
        case_count = length(regexp(str,'case '));
        
        %loop count
        for_count = length(regexp(str,'for '));
        while_count = length(regexp(str,'while '));
        loop_count = for_count + while_count;
        
        %enter key count
        enter_count = length(regexp(str,'"*\n*"'));
        endl_count = length(regexp(str,'endl'));
        enter_key_count = enter_count + endl_count;
        
        %variable count
        int_count = length(regexp(str,'int VAR_'));
        double_count = length(regexp(str,'double VAR_'));
        char_count = length(regexp(str,'char VAR_'));
        float_count = length(regexp(str,'float VAR'));
        
        %sample feature = 19
        sam_fea = [%curr_dir(l).name, ...
                   input_count, ...
                   output_count, ...
                   dotf_count, ...
                   d_count, ...
                   s_count, ...
                   f_count, ...
                   lf_count, ...
                   reference_count, ...
                   antibracket_count, ...
                   if_count, ...
                   switch_count, ...
                   case_count, ...
                   loop_count, ...
                   enter_key_count, ...
                   int_count, ...
                   double_count, ...
                   char_count, ...
                   float_count, ...
                   class];
         tmp(l-2,:) = sam_fea;
    end
    feature(500*(k-3)+1:500*(k-2),:) = tmp;
    fprintf("folder %s is finished...",contains(k).name);
end  