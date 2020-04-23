// note : the first input number should be one cycles later after the reset signal, which is synchronized with the start signal, the 8 output registers update the values right at the 32th bits come in.
// if reset = 1, the output becomes 0 in the same cycle, and idle = 1 at the same cycle as well
// if the size of the img_size reaches, the output_wire will become zero at the end of 32 * img_size th + 33 cycle, since we have to wait for the shift to be finished
module relu_quantizer #(
  parameter SRAM_DEPTH         = 256*256          ,
  parameter SRAM_ADDR_W        = clog2(SRAM_DEPTH),
  parameter START_QUANTIZE_BIT = 11               ,
  parameter END_QUANTIZE_BIT   = 18
) (
  input                        clk         ,
  input                        reset       ,
  input      [SRAM_ADDR_W-1:0] img_size    ,
  input                        data_in     ,
  input                        start       ,
  output                       idle        ,
  output reg [          8-1:0] output_array,
  output                       output_wire
);

  function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
      x_tmp = x-1;
      for(clog2=0; x_tmp>0; clog2=clog2+1) begin
        x_tmp = x_tmp >> 1;
      end
    end
  endfunction

  parameter S_IDEL  = 0;
  parameter S_QUANT = 1;

  wire                   quant_clr     ;
  wire [SRAM_ADDR_W-1:0] cur_cnt_nxt   ;
  wire [            4:0] counter_nxt   ;
  reg                    fsm_state     ;
  reg                    fsm_state_nxt ;
  reg  [SRAM_ADDR_W-1:0] cur_cnt       ;
  wire [            7:0] reg_array_wire;
  wire                   dummy_reg_wire;
  reg  [            7:0] reg_array     ;
  reg                    dummy_reg     ;
  reg                    flag          ;
  reg  [            4:0] counter       ;

  demux1to9 #(.CATCH_START_BIT(START_QUANTIZE_BIT-2)) tt1 (
    .clk       (clk              ),
    .reset     (reset            ),
    .sel       (counter          ),
    .Data_in   (data_in          ),
    .Data_out_0(reg_array_wire[0]),
    .Data_out_1(reg_array_wire[1]),
    .Data_out_2(reg_array_wire[2]),
    .Data_out_3(reg_array_wire[3]),
    .Data_out_4(reg_array_wire[4]),
    .Data_out_5(reg_array_wire[5]),
    .Data_out_6(reg_array_wire[6]),
    .Data_out_7(reg_array_wire[7]),
    .Data_out_8(dummy_reg_wire   )
  );

  assign quant_clr = (cur_cnt == (img_size+2));
  assign idle      = (fsm_state == S_IDEL);
  //assign output_wire = output_array[0];
  assign output_wire = output_array[counter[2:0]];

  always @(posedge clk) begin : proc_fsm_state
    if(reset) begin
      fsm_state <= S_IDEL;
    end else begin
      fsm_state <= fsm_state_nxt;
    end
  end

  assign cur_cnt_nxt = cur_cnt + 1;
  assign counter_nxt = counter + 1;
  always @(posedge clk) begin : proc_cur_cnt
    if(reset) begin
      cur_cnt <= 0;
    end else if (fsm_state == S_IDEL) begin
      counter <= 'b0;
      cur_cnt <= 'b0;
    end

    else if (fsm_state == S_QUANT) begin
      counter <= counter_nxt;
      //output_array <= {1'b0,output_array[7:1]};
      if (counter == 5'b11111) begin
        cur_cnt <= cur_cnt_nxt;
      end
    end

  end


  always @(*) begin : proc_fsm_state_nxt
    fsm_state_nxt = fsm_state;
    case (fsm_state)
      S_IDEL :
        if((start)&&(quant_clr==0)) begin
          fsm_state_nxt = S_QUANT;
        end
      S_QUANT : begin
        if(quant_clr) begin
          fsm_state_nxt = S_IDEL;
        end
      end
    endcase
  end

  always @(posedge clk) begin
    if((fsm_state == S_IDEL) ) begin
      flag         <= 'b0;
      dummy_reg    <= 'b0;
      reg_array    <= 'b0;
      output_array <= 'b0;
    end
    else if((fsm_state == S_QUANT) ) begin
      if((counter > END_QUANTIZE_BIT)&&(counter < 5'b11111)) begin
        if(dummy_reg == 1) begin
          flag <= 1'b1;
        end
      end

      else if(counter == 5'b11111) begin
        if(dummy_reg == 1'b0) begin
          if(flag == 1'b1) begin    // if flag1 is 1 and 32 bits is 0, which means the value is clipped to 6
            output_array <= 8'b11111111;
          end
          else begin
            output_array <= reg_array;
          end
        end
        else begin   // if the value is negative, clip the value to 0
          output_array <= 8'b0;
        end
      end

      else if(counter == 0) begin
        flag <= 0;
      end
      // load the dummy regs
      dummy_reg <= dummy_reg_wire;
      // load the reg_array
      reg_array <= reg_array_wire;
    end
  end
endmodule

