library(dplyr)
library(RColorBrewer)
library(mixtools)
library(scales)
library(ggplot2)
library(tidyr)

# Preprocess grid data with additional group value
pre_grid = function(data){
  # Select columns starting with "x_"
  group_cols <- grep("^x_", names(data), value = TRUE)
  # Create group id by pasting values of those columns together
  data$x <- as.integer(factor(do.call(paste, data[group_cols])))
  # get deviation
  val_cols <- grep("^(f_|t_)", names(data), value = TRUE)
  data <- data %>%
    group_by(x) %>%
    mutate(across(all_of(val_cols), ~ . - median(.), .names = "noise_{.col}")) %>%
    ungroup()
  return(data)
}

# preprocess walk data
pre_walk = function(line_csv){
  d =read.csv(line_csv, sep="\t", header=FALSE, strip.white=TRUE)
  
  colnames(d) = c("f_eval", "f", paste("x_", 0:9, sep=""), "t", "na")
  d = data.frame(d)
  d = subset(d, select=-c(na))
  d = d[complete.cases(d),]
  d$f_eval = as.numeric(d$f_eval)
  ## remove first row because it is duplicate
  d = d[!d$f_eval==1,]
  d <- d %>%
    mutate(walk = cumsum(f_eval == 2))
  return(d)
}

get_line_func_data = function(func, path = "rw-gan-mario-diagonal-walk-random"){
  data = data.frame()
  for(i in 1:7){
    file_name = paste("rw-gan-mario", sprintf("f%03d", func), sprintf("i%02d", i),
                      "d10_rw.txt", sep="_")
    d = pre_walk(paste(path, file_name, sep="/"))
    d$inst = i
    data = rbind(data,d)
  }
  return(data)
}

get_line_data = function(path= "rw-gan-mario-diagonal-walk-random"){
  fs_old = c(1,4,7,10,13,22,25,28,34,37,40,2,5,8,11,14,23,26,29,35,38,41)
  fs = c(1,3,5,7,9,11,17,23,15,21,27,2,4,6,8,10,12,18,24,16,22,28)
  for(i in 1:length(fs)){
    d = get_line_func_data(fs_old[i], path=path)
    if(i==1){
      data = d
    }
    data[paste("f",fs[i], sep="_")] = d$f
    data[paste("t",fs[i], sep="_")] = d$t
  }
  return(data)
}


# predict playability from other values (line plots)
lin_reg = function(data){
  pred = list(o = list(pred= c(1,3,5,7,9), target= c(11,15,17,21,23,27)),
              u = list(pred=c(2,4,6,8,10), target= c(12,16,18,22,24,28)))
  fit_stats = data.frame()
  for(lvl in c("o","u")){
    for(target in pred[[lvl]]$target){
      for(p in pred[[lvl]]$pred){
        plot(data[[paste("f", p, sep="_")]],
             data[[paste("f", target, sep="_")]],
             main=paste(p, "f", target),
             xlab=paste("f", p, sep="_"),
             ylab=paste("f", target, sep="_"))
        plot(data[[paste("f", p, sep="_")]],
             data[[paste("t", target, sep="_")]],
             main=paste(p, "t", target),
             xlab=paste("f", p, sep="_"),
             ylab=paste("t", target, sep="_"))
      }
      #preds_form = paste(paste("f", pred[[lvl]]$pred, sep="_"), collapse = " + ")
      #form = paste("f_", target," ~ ", preds_form, sep="")
      #model = glm(as.formula(form), data=data, family="gaussian")
      #res = summary(model)
      #print(form)
      #plot(model)
      #print(summary(model))
      #d = list(target, res$r.squared, res$adj.r.squared, res$fstatistic[["value"]],
      #      res$fstatistic[["numdf"]], res$fstatistic[["dendf"]])
      #fit_stats=rbind(fit_stats, d)
    }
  }
  #colnames(fit_stats) = c("t", "rsq", "arsq", "fval", "fdf", "fdf2")
  
  
}

# cost distribution for playable vs not playable / fitness
# base r version
plot_fvt = function(data, fs){
  palette <- brewer.pal(length(fs), "Dark2")
  t_cols = paste("t_", fs, sep="")

  par(mar=c(5.1, 4.1, 4.1, 3.1), xpd=TRUE)
  plot(NA, xlim=c(0,1.25),ylim=c(min(log(data[t_cols])), max(log(data[t_cols]))),
       xlab="f", ylab="log t", main="f vs t")
  for(fi in 1:length(fs)){
    points(data[[paste("f_",fs[fi], sep = "")]],
           log(data[[paste("t_",fs[fi],sep="")]]),
           col=palette[fi])
  }
  legend("topright", inset=c(-0.13,0),legend=fs, col=palette, pch=19)
}

#ggplot2 version
plot_fvt_gg <- function(data, fs) {
  # Build the palette with the correct number of colors
  palette <- brewer.pal(length(fs), "Dark2")
  
  # Build column names for f_ and t_
  f_cols <- paste0("f_", fs)
  t_cols <- paste0("t_", fs)
  
  # Select only the columns we need plus any others if needed
  data_subset <- data %>%
    select(all_of(c(f_cols, t_cols)))
  
  # Pivot longer for f and t separately
  f_long <- data_subset %>%
    select(all_of(f_cols)) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "f")
  
  t_long <- data_subset %>%
    select(all_of(t_cols)) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "t")
  
  # Combine f and t together, matching rows by their order and variable name
  plot_data <- f_long %>%
    mutate(variable = sub("^f_", "", variable)) %>%
    bind_cols(t = t_long$t) %>%
    mutate(variable = factor(variable, levels = fs)) %>%
    mutate(log_t = log(t))
  
  # Plot
  p = ggplot(plot_data, aes(x = f, y = log_t, color = variable)) +
    geom_point(alpha = 0.3) + 
    scale_color_manual(values = palette, name = "f series") +
    xlim(0, 1.25) +
    labs(x = "f", y = "log t", title = "f vs t") +
    theme_minimal() +
    theme(
      legend.position = "top",
      legend.title = element_text(face = "bold")
    )
  print(p)
}


  
# cost and fitness distributions multimodal
plot_noise_dist = function(data, fs){
  cuts = seq(0.1,1,0.1)
  dat = matrix(NA, nrow=length(cuts)*length(fs), ncol=3)
  counter = 1
  for(f in fs){
    nf = data[[paste("noise_f_",f,sep="")]] / median(data[[paste("f_",f,sep="")]])
    #plot(data$x, nf, main=f)
    for(cut in cuts){
      d = makemultdata(nf, cuts=c(-cut,cut))
      outliers = (d$y[1] + d$y[3])/d$y[2]
      dat[counter,] = c(outliers,cut,f)
      counter=counter+1
    }

  }
  palette <- brewer.pal(length(fs), "Dark2")
  plot(NA, xlim=c(0.1, 1),ylim=c(0,max(dat[,1])),
       xlab="Outlier relative cuts", ylab="Outlier frequency",
       main="Heavy tails")
  for(fi in 1:length(fs)){
    d = dat[dat[,3]==fs[fi],]
    lines(d[,2],d[,1],col=palette[fi])
  }
  legend("topright", legend=fs, col=palette, lty=1)
}


plot_noise_dist_gg <- function(data, fs, cuts = seq(0.1, 1, 0.1)) {
  
  # Prepare storage for results
  results <- data.frame()
  
  for (f in fs) {
    nf <- data[[paste0("noise_f_", f)]] / median(data[[paste0("f_", f)]], na.rm = TRUE)
    
    for (cut in cuts) {
      d <- makemultdata(nf, cuts = c(-cut, cut))
      outliers <- (d$y[1] + d$y[3]) / d$y[2]
      results <- rbind(results, data.frame(
        outlier_freq = outliers,
        cut = cut,
        feature = f
      ))
    }
  }
  
  # Make feature a factor for color scale
  results$feature <- factor(results$feature, levels = fs)
  
  palette <- brewer.pal(length(fs), "Dark2")
  
  p = ggplot(results, aes(x = cut, y = outlier_freq, color = feature)) +
    geom_line(size = 1) +
    scale_color_manual(values = palette, name = "Feature") +
    labs(
      title = "Heavy tails",
      x = "Outlier relative cuts",
      y = "Outlier frequency"
    ) +
    theme_minimal() +
    theme(legend.position = "top")
  print(p)
}



plot_noise_clean = function(data,fs,cl){
  if(length(cl)>0){
    rm_id = data[,paste("f_",cl,sep="")] ==1
    rm_id = apply(rm_id,1,any)
    rm_x = unique(data$x[rm_id])
    d = data[!(data$x %in% rm_x),]    
  }else{
    d = data
  }
  d <- d %>%
    group_by(x) %>%
    mutate(across(all_of(paste("f_",fs,sep="")),
                  ~ . - median(.), .names = "cn_{.col}")) %>%
    ungroup()
  
  ux = length(unique(data$x))
  
  for(f in fs){
    col_name= paste("cn_f_",f,sep="")
    df <- d %>%
      group_by(x) %>%
      filter(var(across(all_of(col_name))) > 0) %>%
      ungroup()
    df = df %>%
      mutate(nx = as.integer(factor(x)))
    df = df %>%
      group_by(nx) %>%
      count(across(all_of(col_name)), name="freq") %>%
      mutate(total_in_group = sum(freq)) %>%
      mutate(freq_pct = 100*freq / total_in_group)


    uxf = length(unique(df$nx))
    plt = ggplot(df, aes(x = nx, y = !!sym(col_name), color = freq_pct)) +
      geom_point(size=3) +
      scale_x_discrete(drop = TRUE) +
      scale_color_gradient(low = "lightblue", high = "darkblue", name = "Freq (%)") +
      labs(
        x = "sample",
        y = paste("f_",f,sep=""),
        title = paste(uxf,"out of", ux, "samples")
      ) +
      theme_minimal()
    print(plt)
  }
}



# neighbour noise
nnoise = function(data, fs){
  n = length(unique(data$x[data$sample==1]))
  m = length(unique(data$sample))
  for(f in fs){
    nd = matrix(NA, nrow=m, ncol=n)
    fd = nd
    for(i in 1:m){
      sample = as.numeric(levels(factor(data$sample))[i])
      for(j in 1:n){
        x = as.numeric(levels(factor(data$x[data$sample==sample]))[j])
        nvals = data[data$x==x,paste("noise_f_",f,sep="")][[1]]
        nd[i,j] = mean(abs(nvals))
        fvals = data[data$x==x,paste("f_",f,sep="")][[1]]
        fd[i,j] = median(fvals)#sum(fvals)/nrow(fvals)
        
      }
      # gains for resamples
      id = order(nd[i,],decreasing = TRUE)
      ndi = nd[i,id]
      fdi = fd[i,id]
      if(ndi[1]==0){
        nd[i,] =rep(NA,n)
        fd[i,] = rep(NA,n)
      }else{
        val = fdi+ndi >= fdi[1]+ndi[1]
        ndi[which(!val)]=ndi[which(!val)-1]
        fdi[which(!val)] = fdi[which(!val)-1]
        nd[i,] = ndi/abs(fdi)# relative noise
        
      }
    }
    colnames(nd) = 0:10
    boxplot(x=as.list(as.data.frame(nd)),
            main=paste("noise",f),xlab="resamples")
  }
}

nnoise_gg <- function(data, fs) {
  n <- length(unique(data$x[data$sample == 1]))
  m <- length(unique(data$sample))
  
  for (f in fs) {
    nd_list <- list()
    fd_list <- list()
    
    for (i in 1:m) {
      sample_val <- as.numeric(levels(factor(data$sample))[i])
      x_vals <- sort(unique(data$x[data$sample == sample_val]))
      
      nd_vec <- numeric(length = n)
      fd_vec <- numeric(length = n)
      
      for (j in seq_along(x_vals)) {
        x_val <- x_vals[j]
        
        nvals <- data %>% 
          filter(x == x_val) %>% 
          pull(paste0("noise_f_", f))
        
        fvals <- data %>% 
          filter(x == x_val) %>% 
          pull(paste0("f_", f))
        
        nd_vec[j] <- mean(abs(nvals))
        fd_vec[j] <- median(fvals)
      }
      
      # gains for resamples
      id <- order(nd_vec, decreasing = TRUE)
      ndi <- nd_vec[id]
      fdi <- fd_vec[id]
      
      if (ndi[1] == 0) {
        nd_vec[] <- NA
        fd_vec[] <- NA
      } else {
        val <- (fdi + ndi) >= (fdi[1] + ndi[1])
        # Fix indices carefully
        for (k in seq_along(val)) {
          if (!val[k] && k > 1) {
            ndi[k] <- ndi[k - 1]
            fdi[k] <- fdi[k - 1]
          }
        }
        nd_vec[id] <- ndi / abs(fdi)
      }
      
      nd_list[[i]] <- nd_vec
      fd_list[[i]] <- fd_vec
    }
    
    # Combine into a dataframe for ggplot
    nd_df <- do.call(rbind, nd_list)
    colnames(nd_df) <- 0:(n - 1)
    nd_df <- as.data.frame(nd_df)
    nd_df$sample <- factor(1:m)
    
    nd_long <- nd_df %>%
      pivot_longer(cols = -sample, names_to = "resample", values_to = "relative_noise") %>%
      mutate(resample = as.integer(resample))
    
    # Plot with ggplot
    p <- ggplot(nd_long, aes(x = factor(resample), y = relative_noise)) +
      geom_boxplot(outlier.size = 1) +
      labs(
        title = paste("Noise", f),
        x = "Resamples",
        y = "Relative noise (mean abs noise / abs(median f))"
      ) +
      theme_minimal()
    
    print(p)
  }
}


pdf("ngrid.pdf")
ngrid_data = data.frame()
for(exp in c(1,2,3)){
  ngrid_csv = paste("data_dim_10_n_1000_sim_30_exp",exp,".csv", sep="")
  tmp_data = read.csv(ngrid_csv)
  if(nrow(ngrid_data) > 0){
    tmp_data$sample = tmp_data$sample + max(ngrid_data$sample)
  }
  ngrid_data = rbind(ngrid_data, tmp_data)
}
ngrid_csv = "data_dim_10_n_1000_sim_30_exp1.csv"
ngrid_data = read.csv(ngrid_csv)
ngrid_data = pre_grid(ngrid_data)
nnoise(ngrid_data, fs=c(11, 17, 13, 19) )
#nnoise_gg(ngrid_data, fs=c(11, 17, 13, 19) )

#plot_fvt(ngrid_data, fs= c(11, 17, 13, 19))
plot_fvt_gg(ngrid_data, fs= c(11, 17))
plot_fvt_gg(ngrid_data, fs= c(13, 19))

#plot_noise_dist(ngrid_data, fs= c(11, 17, 13, 19))
plot_noise_dist_gg(ngrid_data, fs= c(11, 17, 13, 19))

plot_noise_clean(ngrid_data, fs=c(11,17), cl=c(17))
plot_noise_clean(ngrid_data, fs=c(13,19), cl=c(19))
dev.off()
  

pdf("grid.pdf")
grid_csv = "data_dim_10_n_1000_sim_30_meteor.csv"
grid_data = read.csv(grid_csv)
grid_data = pre_grid(grid_data)
plot_fvt_gg(grid_data, fs= c(11, 17))
plot_fvt_gg(grid_data, fs= c(13, 19))
plot_noise_dist_gg(grid_data, fs= c(11, 17, 13, 19))
plot_noise_clean(grid_data, fs=c(11,17), cl=c(17))
plot_noise_clean(grid_data, fs=c(13,19), cl=c(19))
dev.off()

pdf("linewalk.pdf")
line_data = get_line_data(path="rw-gan-mario-diagonal-walk-random")
#plot_fvt(line_data, fs=c(11,17,23,15,21,27))
#plot_fvt(line_data, fs=c(12,18,24,16,22,28))
plot_fvt_gg(line_data, fs=c(11,17,23))
plot_fvt_gg(line_data, fs=c(15,21,27))
plot_fvt_gg(line_data, fs=c(12,18,24))
plot_fvt_gg(line_data, fs=c(16,22,28))

dev.off()





#f_cols = grep("^f_", names(data), value = TRUE)
#t_cols = grep("^t_", names(data), value = TRUE)
#fs = substr(f_cols, 3, nchar(f_cols))

#mmix = multmixEM(d, k=3)
#cdf <- compCDF(d$x, mmix$posterior,lwd=2, lab=c(7, 5, 7),
#               xlab="Angle in degrees",  ylab="Component CDFs",
#               main="Three-Component Solution")

#nmix = normalmixEM(data[[val]],k=3)
#plot(nmix,whichplots=1)
#plot(nmix,whichplots=2)
#summary(nmix)