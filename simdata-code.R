# This is a code for simulating compartments in single-cell Hi-C data.
# 
# It takes an experimental compartment file and related parameters as input, 
# and generates a file containing multiple sets of simulated data.
#
# The main steps of the simulation include:
# 1. Parameter estimation
# 2. Adding differences
# 3. Resampling
#
# Below is an example that demonstrates the input and output of the function.
process_simulation <- function(num_real_cell, h5_path,r_mean, r_var, r_flag, noise_sd,noise_ratio,sample_N, new_h5_path, group_num, run_num,changetype) {
  # Check if all required parameters are provided
  if (missing(h5_path) || missing(r_mean) || missing(r_var) || missing(r_flag) ||
      missing(noise_sd) || missing(noise_ratio) || missing(sample_N)|| missing(new_h5_path)) {
    cat("Insufficient number of arguments. Please ensure all parameters are provided.")
    return(invisible())
  }
  if (length(r_mean) != length(r_var) || length(r_mean) != length(r_flag)) {
    cat("Error: r_mean, r_var, and r_flag must have the same length.\n")
    return(invisible())
  }
  Ratio_mean <- 0.3
  Ratio_var <- 0.3
  group_id <- 'compartment_raw'
  if (file.exists(new_h5_path)) {
    cat("Result file exists.\n")
  } else {
    cat("Result file does not exist.\n")
    h5createFile(new_h5_path)
  }  
  #estimate parameters
  estimate_parameters <- function(h5_path, group_id) {
    start_time <- Sys.time()
    h5_file <- H5Fopen(h5_path, "H5F_ACC_RDONLY")
    datasets_info <- h5ls(h5_file)
    group <- h5read(h5_file, group_id)
    H5Fclose(h5_file)
    all_cell_ids <- names(group)[grep("^cell_", names(group))]
    total_cell_num <- length(all_cell_ids)
    num_cells_to_read <- min(num_real_cell, total_cell_num)
    cell_id <- paste0('cell_', c(seq(0, num_cells_to_read - 1)))
    raw_df <- data.frame(matrix(nrow = length(group[[cell_id[1]]]), ncol = num_cells_to_read))
    colnames(raw_df) <- cell_id
    for (i in seq_along(cell_id)) {
      raw_df[, i] <- group[[cell_id[i]]]
    }
    end_time <- Sys.time()
    elapsed_time <- end_time - start_time
    print(paste("readtime:",elapsed_time))
    mean_data <- apply(raw_df,1,mean)
    var_data <- apply(raw_df,1,sd)
    return(list(mean_data = mean_data, var_data = var_data))
  }
  result <- estimate_parameters(h5_path, group_id)
  Mean_data <- result$mean_data
  Var_data <- result$var_data
  #初始化标签
  Datalabel2_mean <- rep("nd", length(Mean_data))
  Datalabel2_var <- rep("nd", length(Mean_data))
  Datalabel2_mean_var <- rep("nd", length(Mean_data))
  Datalabel3_mean <- rep("nd", length(Mean_data))
  Datalabel3_var <- rep("nd", length(Mean_data))
  Datalabel3_mean_var <- rep("nd", length(Mean_data))
  simulation_function <- function(r_mean, r_var, r_flag, noise_sd,noise_ratio,sample_N, Ratio_mean, Ratio_var, new_h5_path, mean_data, var_data, group_num, run_num,changetype,num){
    #Add labels
    labelsAB <- rep("", length(mean_data))
    label_block <- rep("", length(mean_data))
    labels <- rep("nd", length(mean_data))
    label_strength <- rep("**", length(mean_data))
    labelsAB<-ifelse(mean_data > 0, 1, 0)
    label_block<-cumsum(abs(c(0, diff(labelsAB))))
    grouped_max <- tapply(mean_data[labelsAB==1], list(labelsAB[labelsAB==1], label_block[labelsAB==1]), max)
    grouped_min <- tapply(mean_data[labelsAB==0], list(labelsAB[labelsAB==0], label_block[labelsAB==0]), min)
    labels3 <- rep(4, length(mean_data))
    A_quantile<-unname(quantile(grouped_max,c(0.25,0.5,0.75)))
    B_quantile<-unname(quantile(abs(grouped_min),c(0.25,0.5,0.75)))
    print(A_quantile)
    print(B_quantile)
    for (i in unique(label_block[labelsAB==1])) {
      max_val <- max(mean_data[label_block==i])
      category <- cut(max_val, breaks = c(-Inf, A_quantile, Inf), labels = c("SA", "MA", "LA", "OA"))
      labels3[label_block == i] <- as.character(as.numeric(category) - 1)
    }
    for (i in unique(label_block[labelsAB==0])) {
      min_val <- abs(min(mean_data[label_block==i]))
      category <- cut(min_val, breaks = c(-Inf, B_quantile, Inf), labels = c("SB", "MB", "LB", "OB"))
      labels3[label_block == i] <- as.character(as.numeric(category) - 1)
    }
    A_quantile_all<-unname(quantile(mean_data[labelsAB == 1],c(0.25,0.5,0.75)))
    B_quantile_all<-unname(quantile(abs(mean_data[labelsAB == 0]),c(0.25,0.5,0.75)))
    print(A_quantile_all)
    print(B_quantile_all)
    label_strength[labelsAB == 1] <- as.character(cut(mean_data[labelsAB == 1], breaks = c(-Inf, A_quantile_all, Inf), labels = c("SA", "MA", "LA", "OA")))
    label_strength[labelsAB == 0] <- as.character(cut(abs(mean_data[labelsAB == 0]), c(-Inf, B_quantile_all, Inf), labels = c("SB", "MB", "LB", "OB")))
    possible_values_A <- label_block[labelsAB == "1" ]
    possible_values_B <- label_block[labelsAB == "0" ]
    possible_values_OA <- label_block[labels3 == "3"&labelsAB == "1" ]
    possible_values_OB <- label_block[labels3 == "3"&labelsAB == "0" ]
    possible_values_LA <- label_block[labels3 == "2"&labelsAB == "1" ]
    possible_values_LB <- label_block[labels3 == "2"&labelsAB == "0" ]
    possible_values_MA <- label_block[labels3 == "1"&labelsAB == "1" ]
    possible_values_MB <- label_block[labels3 == "1"&labelsAB == "0" ]
    possible_values_SA <- label_block[labels3 == "0"&labelsAB == "1" ]
    possible_values_SB <- label_block[labels3 == "0"&labelsAB == "0" ]
    select_data <- function(my_array, target_percentage) {
      len <- length(my_array)
      selected_data <- c()  
      total_selected <- 0  
      if(len==0) return(selected_data)
      while (total_selected / len < target_percentage) {
        selected_label <- sample(unique(my_array), size = 1)
        selected_group <- my_array[my_array == selected_label]
        selected_data <- c(selected_data, selected_group)
        total_selected <- length(selected_data)
        my_array <- my_array[!my_array %in% selected_label]  
      }
      return(selected_data)
    }
    if (r_mean != 1) {
      target_percentage <- Ratio_mean
    } else if (r_var != 1) {
      target_percentage <- Ratio_var
    } else {
      target_percentage <- 1
    }
    seeds1 <- c(123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066, 7077)  
    seed1 <- seeds1[r_flag]
    set.seed(seed1)  
    random_values_OA <- select_data(possible_values_OA, target_percentage)
    random_values_OB <- select_data(possible_values_OB, target_percentage)
    random_values_LA <- select_data(possible_values_LA, target_percentage)
    random_values_LB <- select_data(possible_values_LB, target_percentage)
    random_values_MA <- select_data(possible_values_MA, target_percentage) 
    random_values_MB <- select_data(possible_values_MB, target_percentage) 
    random_values_SA <- select_data(possible_values_SA, target_percentage)
    random_values_SB <- select_data(possible_values_SB, target_percentage) 
    mean_data_new <- mean_data 
    var_data_new <- var_data 
    if (r_mean != 1) {
      for (i in 1:length(mean_data)) {
        if (label_block[i] %in% c(random_values_OA, random_values_LA, random_values_MA, random_values_SA,random_values_OB, random_values_LB, random_values_MB, random_values_SB)){
          mean_data_new[i] <- mean_data[i] * r_mean
          labels[i] <- 'wd'
        }
      }
    }
    if(r_mean==1&&r_var!=1){
      for (i in 1:length(var_data)) {
        if (label_block[i] %in% c(random_values_OA, random_values_LA, random_values_MA, random_values_SA,random_values_OB, random_values_LB, random_values_MB, random_values_SB)) {
          var_data_new[i] <- var_data[i] * r_var
          labels[i] <- 'vd'
        }
      }
    }
    seeds2 <-  c(123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066, 7077)   # 十组随机种子
    seed2 <- seeds2[r_flag]
    set.seed(seed2)  
    select_var_data <- function(my_array,len, target_percentage) {
      selected_var_data <- c()  
      total_selected <- 0  
      if(len==0) return(selected_var_data)
      while (total_selected / len < target_percentage) {
        selected_label <- sample(unique(my_array), size = 1)
        selected_group <- my_array[my_array == selected_label]
        selected_var_data <- c(selected_var_data, selected_group)
        total_selected <- length(selected_var_data)
        my_array <- my_array[!my_array %in% selected_label] 
      }
      
      return(selected_var_data)
    }
    possible_values_varA <- label_block[labelsAB == "1" &labels == "nd"]
    possible_values_varB <- label_block[labelsAB == "0" &labels == "nd"]
    lenA <- length(possible_values_A)
    lenB <- length(possible_values_B)
    random_varA <- select_var_data(possible_values_varA, lenA, Ratio_var)
    random_varB <- select_var_data(possible_values_varB, lenB, Ratio_var)
    if(r_mean!=1&&r_var!=1){
      for (i in 1:length(var_data)) {
        if (label_block[i] %in% c(random_varA, random_varB)) {
          var_data_new[i] <- var_data[i] * r_var
          labels[i] <- 'vd'
        }
      }
    }
    #Check the number and proportion of labels
    label_proportions <- table(labels)
    labels3_proportions <- table(labels3)
    labelSTR_proportions <- table(label_strength)
    cat("simulation label:\n")
    cat("notion: the label is nd、wd、vd\n")
    print(label_proportions)
    cat("Experimental data partitioning situation:\n")
    cat("notion: the regional strength is 0-3\n")
    print(labels3_proportions)
    cat("notion: the Overall strength is 0-3\n")
    print(labelSTR_proportions)
    datalabel <- data.frame(
      labelsAB,
      labels3
    )
    table_label1 <- table(datalabel$labels3[datalabel$labelsAB == '1'])
    table_label0 <- table(datalabel$labels3[datalabel$labelsAB == '0'])
    cat("Experimental data partitioning situation in A compartment:\n")
    print(table_label1)
    cat("Experimental data partitioning situation in B compartment:\n")
    print(table_label0)
    num_cores <- detectCores()
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    sampled_data_list <- foreach(i = 1:length(mean_data)) %dopar% {
      mu_param <- mean_data_new[i]
      sigma_param <- var_data_new[i]
      set.seed(num+i) 
      sampled_values <- rnorm(sample_N, mean = mu_param, sd = sigma_param) 
      return(sampled_values)
    }
    stopCluster(cl)
    gc()
    #Add noise
    sampled_data_list <- matrix(unlist(sampled_data_list), nrow = length(sampled_data_list), byrow = TRUE)
    nrow <- nrow(sampled_data_list)  
    ncol <- ncol(sampled_data_list) 
    num_elements <- nrow * ncol
    set.seed(num)  
    num_replace <- ceiling(noise_ratio * num_elements)  
    sampled_data_new <- matrix(0, nrow = nrow, ncol = ncol)
    replace_indices <- sample(num_elements, num_replace, replace = FALSE)
    sampled_data_new[replace_indices] <- rnorm(num_replace, mean = 1, sd = 1)  
    sampled_data_list <- sampled_data_list + noise_sd*sampled_data_new 
    cat("write simulation data\n")
    sampled_data_df <- data.frame(sampled_data_list)
    colnames(sampled_data_df) <- paste("cell_", 0:(ncol(sampled_data_df)-1), sep = "")
    sampled_data_df$bin <- 0:(nrow(sampled_data_df)-1)
    sampled_data_df <- sampled_data_df[, c("bin", setdiff(names(sampled_data_df), "bin"))]
    sampled_data_df <- cbind(sampled_data_df,Meanraw = mean_data,Varianceraw=var_data,Meannew = mean_data_new,Variancenew=var_data_new,datalabel = labels,STRlabel = label_strength)
    all(sapply(sampled_data_df, is.vector))
    if(group_num==1){
      group_path <- paste0("mean",r_mean,"_var",r_var,"_noise",noise_sd,"_group1")
    }
    if(group_num==2){
      group_path <- paste0("mean_var_noise_group2")
    }
    if(group_num > 2){
      group_path <- paste0("mean_var_group",group_num)
    }
    objects <- h5ls(new_h5_path)
    if (paste0("/",group_path) %in% objects$group) {
      print(paste("folder", paste0("/",group_path), "exist"))
    } else {
      print(paste("folder", paste0("/",group_path), "does not exist"))
      h5createGroup(new_h5_path, group_path)
    }
    if(changetype==2){
      if(group_num!=2){output_groupname <-  paste0("mean", r_mean,"_flag", r_flag,"_group",group_num,"_run",run_num)}
      if(group_num==2){output_groupname <-  paste0("mean", r_mean,"_group",group_num,"_run",run_num)} }
    if(changetype==3){
      if(group_num!=2){output_groupname <-  paste0("var", r_var,"_flag", r_flag,"_group",group_num,"_run",run_num)}
      if(group_num==2){output_groupname <-  paste0("var", r_var,"_group",group_num,"_run",run_num)} }
    if(changetype==4){output_groupname <-  paste0("noise", noise_sd,"_mean", r_mean, "_group",group_num,"_run",run_num) }
    if(changetype==5){output_groupname <-  paste0("noise", noise_sd,"_var", r_var, "_group",group_num,"_run",run_num) }
    if(changetype==6){output_groupname <-  paste0("mean", r_mean, "+var", r_var,"_group",group_num,"_run",run_num) }
    if(group_num!=1){
    if (output_groupname %in% objects$name) {
      print(paste("file", output_groupname, "exist"))
    } else {
      print(paste("file", output_groupname, "does not exist"))
      h5createGroup(new_h5_path, paste0("/",group_path,"/",output_groupname))
    }}
    for (col_name in colnames(sampled_data_df)) {
      if(group_num!=1)
      {h5write(sampled_data_df[[col_name]], new_h5_path, paste0("/",group_path,"/",output_groupname, "/",col_name))}
      if(group_num==1)
        {h5write(sampled_data_df[[col_name]], new_h5_path, paste0("/",group_path, "/",col_name))}
    }
    h5closeAll()
    return(list(datalabel=labels,dateSTR=label_strength))
  }
  create_merged_vector <- function(vector1, vector2, placeholder = 'nd') {
    vector3 <- character(length(vector1))  
    for (i in 1:length(vector1)) {
      if (vector1[i] != placeholder) {
        vector3[i] <- vector1[i]
      } else if (vector2[i] != placeholder) {
        vector3[i] <- vector2[i]
      } else {
        vector3[i] <- placeholder  
      }
    }
    return(vector3)
  }
  start_time <- Sys.time()
  for (i in 1:length(r_mean)) {
    resultlabel <- simulation_function(r_mean[i], r_var[i], r_flag[i], noise_sd[i],noise_ratio[i],sample_N, Ratio_mean, Ratio_var, new_h5_path, Mean_data, Var_data, group_num[i],run_num[i],changetype[i],i)
    if (changetype[i] == 2&group_num[i] > 2 ) {
      Datalabel3_mean <- create_merged_vector(Datalabel3_mean, resultlabel$datalabel)
    } 
    if (changetype[i] == 3&group_num[i] > 2) {
      Datalabel3_var <- create_merged_vector(Datalabel3_var, resultlabel$datalabel)
    } 
    if (changetype[i] == 2&group_num[i] == 2) {
      Datalabel2_mean <- resultlabel$datalabel
    } 
    if (changetype[i] == 3&group_num[i] == 2) {
      Datalabel2_var <- resultlabel$datalabel
    }
    if (group_num[i] == 4&group_num[i] == 2) {
      Datalabel2_mean <- resultlabel$datalabel
    } 
    if (group_num[i] == 5&group_num[i] == 2) {
      Datalabel2_var <- resultlabel$datalabel
    } 
    if (changetype[i] == 6&group_num[i] == 2) {
      Datalabel2_mean_var <- resultlabel$datalabel
    } 
    dateSTR <- resultlabel$dateSTR
  }
  #Generate simulation data information table
  bin_id <- c(seq(0, length(Datalabel2_mean) - 1))
  csv_file <- paste0(tools::file_path_sans_ext(new_h5_path),"_info.csv")
  data <- data.frame(bin = bin_id, Datalabel_TWO_mean = Datalabel2_mean,Datalabel_TWO_var = Datalabel2_var, Datalabel_MUL_mean = Datalabel3_mean,Datalabel_MUL_var = Datalabel3_var,dateSTR = dateSTR )
  write.csv(data, file = csv_file, row.names = FALSE)
  end_time <- Sys.time()
  elapsed_time <- end_time - start_time
  print(paste("processing time:",elapsed_time))
}
library(rhdf5)
library(progress)
library(parallel)
library(doParallel)
#Original data path
h5_path <- '/data/schic_data/raw/scCompartment.hdf5'
r_mean <- c(1,0.25,0.5,2,4,1,1,1,1,0.25,0.25,1,1)
r_var <- c(1,1,1,1,1,0.25,0.5,2,4,1,1,0.25,0.25)
#Generate multiple sets of wd and vd with different identifiers at different positions, with values ranging from 0 to 9
r_flag <- c(1,1,1,1,1,1,1,1,1,2,3,2,3)
#1 represents no change,
#2 represents the change in mean foldchange, 
#3 represents the change in variance foldchange, 
#4 represents the change in noise intensity, with a fixed change in mean foldchange, 
#5 represents the change in noise intensity, with a fixed change in variance foldchange, 
changetype <- c(1,2,2,2,2,3,3,3,3,2,2,3,3) 
#Noise intensity
noise_sd <- c(0,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08)
#Noise ratio
noise_ratio <- 0.1*c(1,1,1,1,1,1,1,1,1,1,1,1,1)
#group_num=1 indicates that this set of data is generated through original parameter sampling.
#group_num=2 indicates that this set of data is input into the model for differential analysis along with group1.
#group_num=3 indicates that this set of data is input into the model for differential analysis together with other data from the same run_num.
#group_num and run_num represent the data groupings and corresponding run numbers, with different groupings reflecting different experimental designs.
group_num <- c(1,2,2,2,2,2,2,2,2,3,3,3,3)
run_num <- c(1,1,2,3,4,5,6,7,8,1,1,2,2)
#Number of cells in the experimental data
num_real_cell  <- 1171
#Number of cells in simulated data
sample_N  <- 1000
#Store file path
new_h5_path <-'/simdata.hdf5'
process_simulation(num_real_cell, h5_path,r_mean, r_var,r_flag,noise_sd,noise_ratio, sample_N, Ratio_mean, Ratio_var, new_h5_path, group_num, run_num,changetype)


